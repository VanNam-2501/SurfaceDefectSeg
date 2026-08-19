# Aluminum Surface Lab Demo

This local web demo loads the three trained segmentation architectures. Its
default production decision is the fully automatic U-Net + VMamba hybrid that
was selected on Validation and reported once on Test. SegFormer remains
available as an individually reported experiment, while the frozen hybrid
decision uses exactly U-Net and VMamba. Each active model provides its overlay
and mask.

## Before the first run

1. Keep the final checkpoints and decision artifacts in their documented
   workspace locations, or override them as described in
   [INFERENCE_SETUP.md](./INFERENCE_SETUP.md).
2. Install the backend requirements from `backend/requirements.txt`.
3. Start the API with `uvicorn backend.app:app --reload --port 8000` from this
   directory.
4. Start the web interface with `npm run dev`, then open the local URL shown.

The frontend never fabricates predictions: a model is marked unavailable until
its checkpoint is present. With a frozen policy, only policy-compatible models
are selectable. VMamba also needs its compatible runtime and CUDA setup from
the main experiment package.
