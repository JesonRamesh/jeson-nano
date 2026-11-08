/** Browser full screen for the whole page (the dashboard's F key, the Start screen's button). */
export function toggleFullscreen(): void {
  if (document.fullscreenElement) void document.exitFullscreen()
  else void document.documentElement.requestFullscreen()
}
