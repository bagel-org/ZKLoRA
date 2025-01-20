import argparse

from zklora import BaseModelClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_a", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--combine_mode", choices=["replace","add_delta"], default="add_delta")
    parser.add_argument("--verify_proofs", action="store_true")
    args = parser.parse_args()

    client = BaseModelClient(args.base_model, args.host_a, args.port_a, args.combine_mode)
    client.init_and_patch()

    # forward pass => triggers submodule calls => A accumulates => .onnx => proofs
    text = "Hello World, this is a LoRA test."

    loss_val = client.forward_loss(text)
    print(f"[B] final loss => {loss_val:.4f}")

    client.end_inference()
    print("[B] done. B can now fetch proof files from A and verify proofs.")

if __name__=="__main__":
    main()