# Aluminum Surface Lab Demo

This local web demo loads the three trained segmentation architectures. It
offers three original raw baselines, three Adaptive single-model policies,
and three Spatial two-model ensembles. Raw modes use each model's
frozen Validation threshold without component filtering; Adaptive single modes load the component policy and Spatial pair modes load
their matching spatial policy. Each active model provides its overlay and
mask.

## Before the first run

1. Keep the final checkpoints and decision artifacts in their documented
   workspace locations, or override them as described in
   [INFERENCE_SETUP.md](./INFERENCE_SETUP.md).
2. Install the backend requirements from `backend/requirements.txt`.
3. Start the API with `uvicorn backend.app:app --reload --port 8000` from this
   directory.
4. Start the web interface with `npm run dev`, then open the local URL shown.

The frontend never fabricates predictions: a mode is marked unavailable until
all required checkpoints and, for Adaptive/Spatial modes, the matching frozen policy
are present. VMamba also needs its compatible runtime and CUDA setup from the
main experiment package.
