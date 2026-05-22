import { IModelGraph } from '../core/IR';
import { Toast } from '../ui/Toast';

/**
 * 580, 581. Encrypt weight tensors natively using AES-GCM via WebCrypto.
 * Allows requiring a passphrase to decrypt and execute protected models.
 */
export class TensorEncryption {
  private static async getKeyMaterial(password: string): Promise<CryptoKey> {
    const enc = new TextEncoder();
    return window.crypto.subtle.importKey('raw', enc.encode(password), { name: 'PBKDF2' }, false, [
      'deriveBits',
      'deriveKey',
    ]);
  }

  private static async deriveKey(passwordKey: CryptoKey, salt: Uint8Array): Promise<CryptoKey> {
    return window.crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: salt,
        iterations: 100000,
        hash: 'SHA-256',
      },
      passwordKey,
      { name: 'AES-GCM', length: 256 },
      true,
      ['encrypt', 'decrypt'],
    );
  }

  public static async encryptModel(model: IModelGraph, password: string): Promise<IModelGraph> {
    const salt = window.crypto.getRandomValues(new Uint8Array(16));
    const passKey = await this.getKeyMaterial(password);
    const aesKey = await this.deriveKey(passKey, salt);

    // Deep clone schema
    const clonedGraph: IModelGraph = JSON.parse(JSON.stringify(model));

    // Encrypt initializers in place
    for (let i = 0; i < clonedGraph.initializers.length; i++) {
      const init = clonedGraph.initializers[i];
      const originalData = model.initializers[i].rawData;

      if (originalData) {
        const iv = window.crypto.getRandomValues(new Uint8Array(12));
        const encryptedBuf = await window.crypto.subtle.encrypt(
          { name: 'AES-GCM', iv: iv },
          aesKey,
          originalData.buffer,
        );

        // Prepend IV to cipher payload
        const combined = new Uint8Array(12 + encryptedBuf.byteLength);
        combined.set(iv, 0);
        combined.set(new Uint8Array(encryptedBuf), 12);

        init.rawData = combined;
      }
    }

    const docMeta = clonedGraph.docString ? JSON.parse(clonedGraph.docString) : {};
    docMeta.encrypted = true;
    docMeta.salt = Array.from(salt);
    clonedGraph.docString = JSON.stringify(docMeta);

    Toast.show('Model weights encrypted successfully (AES-GCM)', 'success');
    return clonedGraph;
  }

  public static async decryptModel(model: IModelGraph, password: string): Promise<IModelGraph> {
    let docMeta;
    try {
      docMeta = model.docString ? JSON.parse(model.docString) : {};
    } catch (e) {
      throw new Error('Model missing valid metadata for decryption.');
    }

    if (!docMeta.encrypted || !docMeta.salt) {
      throw new Error('Model is not flagged as encrypted.');
    }

    const salt = new Uint8Array(docMeta.salt);
    const passKey = await this.getKeyMaterial(password);
    const aesKey = await this.deriveKey(passKey, salt);

    const clonedGraph: IModelGraph = JSON.parse(JSON.stringify(model));

    // Decrypt initializers in place
    // 582. Execute decrypted portions strictly in WASM buffers (staged for AOT hook)
    for (let i = 0; i < clonedGraph.initializers.length; i++) {
      const init = clonedGraph.initializers[i];
      const encryptedData = model.initializers[i].rawData;

      if (encryptedData) {
        const iv = encryptedData.slice(0, 12);
        const cipherText = encryptedData.slice(12);

        try {
          const decryptedBuf = await window.crypto.subtle.decrypt(
            { name: 'AES-GCM', iv: iv },
            aesKey,
            cipherText.buffer,
          );
          init.rawData = new Uint8Array(decryptedBuf);
        } catch (e) {
          throw new Error(
            `Decryption failed on tensor ${init.name}. Invalid password or corrupt data.`,
          );
        }
      }
    }

    docMeta.encrypted = false;
    delete docMeta.salt;
    clonedGraph.docString = JSON.stringify(docMeta);

    Toast.show('Model weights decrypted successfully', 'success');
    return clonedGraph;
  }
}
