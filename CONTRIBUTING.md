# Contributing

Open an issue for a reproducible defect or a concrete proposed capability. Include a minimal nonconfidential input and the package version. Do not upload private calculation data or credentials.

Create a focused pull request with tests demonstrating observable behavior. Install from the repository and run the commands in the README. Public contracts must remain consistent with the producer and other readers; format changes require an explicit shared contract revision. Numerical acceptance must not be inferred from a successful process or export.

Integration CI runs in the [qcl-negf superproject](https://github.com/Afonenko-QCL-NEGF/qcl-negf) on its isolated local runner. Update the component gitlink there to test a selected change with the complete dependency graph. Maintainers review contributions before running them with access to laboratory infrastructure. Source code is MIT licensed.
