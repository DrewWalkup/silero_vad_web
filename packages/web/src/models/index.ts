import { OrtConfigurer } from "./common"

export * from "./common"
export { SileroLegacy } from "./legacy"
export { SileroV5 } from "./v5"
export { SileroV6 } from "./v6"

export type OrtOptions = {
  ortConfig?: OrtConfigurer
}
