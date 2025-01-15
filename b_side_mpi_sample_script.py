import argparse

from zklora import BaseModelClient, batch_verify_proofs

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

    out_dir = client.end_inference_and_retrieve_proofs()
    print("[B] local proofs =>", out_dir)

    if args.verify_proofs:
        print("[B] verifying proofs in", out_dir)
        total_t, c = batch_verify_proofs(proof_dir=out_dir, verbose=True)
        print(f"[B] verified {c} proof(s) in {total_t:.2f} sec.")

if __name__=="__main__":
    main()