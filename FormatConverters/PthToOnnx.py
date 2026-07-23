"""
.pth (PyTorch) → ONNX 변환기

models/PytorchModels/*.pth 는 상태사전(state_dict)이 아니라 torch.save(model)로
직렬화된 전체 모델 객체다 (예: onnx2pytorch.ConvertModel — 원래 ONNX였던 모델을
pruning 등 학습 파이프라인에서 다루기 위해 PyTorch로 역변환해둔 것). 그래서 이
변환기는 별도의 아키텍처 정의 없이 torch.load로 모델 객체를 그대로 복원한 뒤
torch.onnx.export로 다시 ONNX 그래프를 뽑아낸다.

state_dict만 저장된 .pth(아키텍처 정보 없이 가중치 텐서만 있는 dict)는 이
스크립트로 변환할 수 없다 — 원본 nn.Module 클래스가 있어야 한다.

사용법
------
    python FormatConverters/PthToOnnx.py <input.pth> [output.onnx] [--input-size N] [--opset N]

    input.pth   : 변환할 PyTorch 모델 경로 (torch.save(model)로 저장된 전체 모델 객체, 필수)
    output.onnx : 출력 경로 (생략 시 Onnx/<input파일명>.onnx 에 저장)
    --input-size: 입력 벡터 크기. 생략 시 모델의 첫 번째 nn.Linear 레이어의
                  in_features로 자동 추론한다 (완전연결망 전용 추론이므로, 추론이
                  틀리거나 실패하면 직접 지정할 것).
    --opset     : ONNX opset 버전 (기본값 13)

예제
----
    python FormatConverters/PthToOnnx.py "PytorchModels/model.pth"
    → Onnx/model.onnx 생성 (입력 크기는 자동 추론)

    python FormatConverters/PthToOnnx.py "PytorchModels/model.pth" "Onnx/my_model.onnx" --input-size 256
"""

import argparse
import os
import sys

import numpy as np
import onnx
import torch


def load_model(input_path):
    try:
        obj = torch.load(input_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"'{input_path}' 로드에 필요한 패키지가 없습니다 ({e}). "
            f"이 .pth가 onnx2pytorch로 만든 모델이라면 `pip install onnx2pytorch`를 실행하세요."
        ) from e

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        nested = next((v for v in obj.values() if isinstance(v, torch.nn.Module)), None)
        if nested is not None:
            model = nested
        else:
            raise ValueError(
                f"'{input_path}'는 nn.Module 객체가 아니라 dict(아마 state_dict)입니다. "
                f"이 변환기는 torch.save(model)로 저장된 전체 모델 객체만 지원합니다 "
                f"(state_dict만 있는 경우 원본 모델 클래스로 직접 로드한 뒤 변환해야 합니다)."
            )
    else:
        raise TypeError(f"'{input_path}'에서 지원하지 않는 타입을 로드했습니다: {type(obj)}")

    model.eval()
    return model


def infer_input_size(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return module.in_features
    raise ValueError(
        "모델에서 nn.Linear 레이어를 찾지 못해 입력 크기를 자동 추론할 수 없습니다. "
        "--input-size 옵션으로 직접 지정하세요."
    )


def convert(input_path, output_path=None, input_size=None, opset=13, verify=True):
    if output_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        onnx_dir = os.path.join(project_root, "Onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(onnx_dir, base + ".onnx")

    print(f"로딩 중: {input_path}")
    model = load_model(input_path)

    if input_size is None:
        input_size = infer_input_size(model)
        print(f"입력 크기 자동 추론: {input_size} (첫 번째 nn.Linear 기준)")
    else:
        print(f"입력 크기: {input_size}")

    dummy_input = torch.zeros(1, input_size, dtype=torch.float32)
    try:
        with torch.no_grad():
            torch_output = model(dummy_input)
    except Exception as e:
        raise RuntimeError(
            f"입력 크기 {input_size}로 forward 실행에 실패했습니다: {e}\n"
            f"--input-size 옵션으로 올바른 입력 크기를 직접 지정하세요."
        ) from e

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"변환 완료: {output_path}")
    print(f"출력 크기: {torch_output.shape[-1]}")

    if verify:
        _verify(output_path, model, input_size, torch_output)

    return output_path


def _verify(onnx_path, model, input_size, reference_output):
    try:
        import onnxruntime as ort
    except ImportError:
        print("참고: onnxruntime이 없어 수치 검증을 건너뜁니다 (`pip install onnxruntime`)")
        return

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for _ in range(5):
        x = np.random.randn(1, input_size).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {"input": x})[0]
        max_diff = max(max_diff, float(np.abs(expected - actual).max()))

    status = "OK" if max_diff < 1e-4 else "경고: 오차가 큽니다"
    print(f"수치 검증 ({status}): PyTorch 출력과 ONNX 출력의 최대 절댓값 오차 = {max_diff:.3e}")


def main():
    parser = argparse.ArgumentParser(description=".pth → ONNX 변환기")
    parser.add_argument("input_path", help="변환할 .pth 파일 경로")
    parser.add_argument("output_path", nargs="?", default=None, help="출력 .onnx 경로 (생략 가능)")
    parser.add_argument("--input-size", type=int, default=None, help="입력 벡터 크기 (생략 시 자동 추론)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset 버전 (기본값 13)")
    parser.add_argument("--no-verify", action="store_true", help="onnxruntime 수치 검증 생략")
    args = parser.parse_args()

    convert(
        args.input_path,
        args.output_path,
        input_size=args.input_size,
        opset=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python PthToOnnx.py <input.pth> [output.onnx] [--input-size N] [--opset N]")
        sys.exit(1)
    main()
