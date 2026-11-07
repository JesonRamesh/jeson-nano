/** The UI never shows em dashes (team style). Text from the sim (trace reasons, Gemma
 *  captions, guidance) goes through this before it's displayed. */
export function noEmDash(s: string): string {
  return s.replace(/\s*\u2014\s*/g, ', ')
}
