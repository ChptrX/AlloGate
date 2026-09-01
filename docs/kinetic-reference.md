# Fixed kinetic-reference contract

`KineticReference` is a portable radial-basis projection from learned collective-variable coordinates to a fixed target in `[0, 1]`. It contains only:

- centers in normalized CV space;
- one target value per center;
- the CV mean and whitening transform;
- a positive radial bandwidth;
- a lag value and portable unit token.

Its semantic digest covers every scientific array and parameter. It does not contain training paths, topology identifiers, checkpoints, host information, or runtime devices.

The same reference can be evaluated with NumPy or with differentiable float64 PyTorch arithmetic. Saving and loading verifies the semantic digest. Construction of the reference—feature selection, slow-coordinate training, state discretization, boundary choice, and kinetic validation—remains a separate, method-declared workflow so that a fitted target cannot be confused with Gate evaluation.
