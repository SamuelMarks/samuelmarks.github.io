# Netron UI / Model Modifier

> **Ecosystem Context:** The `onnx9000` ecosystem includes a powerful, web-native visual modifier tool derived from the popular Netron architecture, allowing you to edit ONNX models interactively.

This guide covers the usage of the Netron UI (Model Modifier).

## Overview

While tools like Netron are traditionally used as _viewers_ for neural network, `onnx9000` extends this paradigm to allow _editing_. The Model Modifier UI allows you to inspect, prune, rename, and alter the structure of an ONNX graph directly in a local web interface.

## Usage

### CLI

You can launch the Model Modifier UI using the `edit` command:

```bash
onnx9000 edit model.onnx
```

This will spin up a local development server (typically using Vite in the `apps/netron-ui` package) and open the graph in your default web browser.

### Features

- **Interactive Pruning:** Click on nodes and press `Delete` to remove them from the graph.
- **Node Inspection:** View attributes, inputs, outputs, and tensor shapes.
- **Save Changes:** Export the modified ONNX graph directly from the browser interface back to your local file system.
- **Schema Validation:** The UI integrates with the `onnx9000-core` checker to ensure that manual edits do not produce an invalid ONNX schema.
