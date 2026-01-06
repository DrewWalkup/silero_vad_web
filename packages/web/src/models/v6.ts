import * as ort from "onnxruntime-web/wasm"
import { log } from "../logging"
import { ModelFactory, ModelFetcher, SpeechProbabilities } from "./common"

const CONTEXT_SIZE = 64 // Context size for 16kHz sample rate

function getNewState(ortInstance: typeof ort) {
  const zeroes = Array(2 * 128).fill(0)
  return new ortInstance.Tensor("float32", zeroes, [2, 1, 128])
}

function getNewContext() {
  return new Float32Array(CONTEXT_SIZE)
}

export class SileroV6 {
  private _context: Float32Array

  constructor(
    private _session: ort.InferenceSession,
    private _state: ort.Tensor,
    private _sr: ort.Tensor,
    private ortInstance: typeof ort
  ) {
    this._context = getNewContext()
  }

  static new: ModelFactory = async (
    ortInstance: typeof ort,
    modelFetcher: ModelFetcher
  ) => {
    log.debug("Loading VAD...")
    const modelArrayBuffer = await modelFetcher()
    const _session = await ortInstance.InferenceSession.create(modelArrayBuffer)

    const _sr = new ortInstance.Tensor("int64", [16000n])
    const _state = getNewState(ortInstance)
    log.debug("...finished loading VAD")
    return new SileroV6(_session, _state, _sr, ortInstance)
  }

  reset_state = () => {
    this._state = getNewState(this.ortInstance)
    this._context = getNewContext()
  }

  process = async (audioFrame: Float32Array): Promise<SpeechProbabilities> => {
    // Concatenate context with audio frame (context_size + frame_size = 64 + 512 = 576)
    const inputWithContext = new Float32Array(CONTEXT_SIZE + audioFrame.length)
    inputWithContext.set(this._context, 0)
    inputWithContext.set(audioFrame, CONTEXT_SIZE)

    const t = new this.ortInstance.Tensor("float32", inputWithContext, [
      1,
      inputWithContext.length,
    ])
    const inputs = {
      input: t,
      state: this._state,
      sr: this._sr,
    }
    const out = await this._session.run(inputs)

    if (!out["stateN"]) {
      throw new Error("No state from model")
    }
    this._state = out["stateN"] as ort.Tensor

    // Update context with last CONTEXT_SIZE samples from input
    this._context = inputWithContext.slice(-CONTEXT_SIZE)

    if (!out["output"]?.data) {
      throw new Error("No output from model")
    }
    const [isSpeech] = out["output"]?.data as unknown as [number]
    const notSpeech = 1 - isSpeech
    return { notSpeech, isSpeech }
  }
}
