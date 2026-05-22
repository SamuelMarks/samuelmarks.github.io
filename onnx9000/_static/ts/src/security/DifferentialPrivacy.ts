/**
 * Implements Differential Privacy (DP) mechanisms via WebCrypto API.
 * Tasks 572, 573, 574
 */

export class DifferentialPrivacy {
  private epsilon: number;
  private delta: number;
  private sensitivity: number;

  constructor(epsilon = 1.0, delta = 1e-5, sensitivity = 1.0) {
    this.epsilon = epsilon;
    this.delta = delta;
    this.sensitivity = sensitivity;
  }

  /**
   * Generates Gaussian noise securely using WebCrypto.
   * Standard Box-Muller transform applied to uniform values from crypto.getRandomValues.
   */
  private generateSecureGaussianNoise(): number {
    const u = new Uint32Array(2);
    window.crypto.getRandomValues(u);

    // Convert to [0, 1) safely
    const u1 = u[0] / (0xffffffff + 1);
    const u2 = u[1] / (0xffffffff + 1);

    // Box-Muller
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0;
  }

  /**
   * 574. Injects DP noise into a flat Float32Array (e.g. Gradients).
   */
  public injectNoise(gradients: Float32Array): Float32Array {
    const noisyGradients = new Float32Array(gradients.length);

    // Calculate Gaussian mechanism scale (sigma)
    // sigma = sqrt(2 * log(1.25 / delta)) * sensitivity / epsilon
    const sigma = (Math.sqrt(2.0 * Math.log(1.25 / this.delta)) * this.sensitivity) / this.epsilon;

    for (let i = 0; i < gradients.length; i++) {
      const noise = this.generateSecureGaussianNoise() * sigma;
      noisyGradients[i] = gradients[i] + noise;
    }

    return noisyGradients;
  }
}
