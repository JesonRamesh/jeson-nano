import { useId } from 'react'

/**
 * Four-pointed spark in the Gemini colours, the dashboard's sign for Gemma.
 * active: Gemma is working on it right now (pulses, glows, colours turn); otherwise a quiet mark.
 */
export function GeminiSpark({ active = false, size = 14, title }: { active?: boolean; size?: number; title?: string }) {
  const id = useId().replace(/:/g, '')
  return (
    <svg
      className={`gspark ${active ? 'active' : ''}`}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role="img"
      aria-label={title ?? (active ? 'Gemma thinking' : 'Gemma')}
    >
      {title && <title>{title}</title>}
      <defs>
        <linearGradient id={`g${id}`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#3a8bff" />
          <stop offset="0.45" stopColor="#3a8bff" />
          <stop offset="0.68" stopColor="#ee5257" />
          <stop offset="0.84" stopColor="#d0c42e" />
          <stop offset="1" stopColor="#4cc88b" />
          {active && (
            <animateTransform
              attributeName="gradientTransform"
              type="rotate"
              from="0 0.5 0.5"
              to="360 0.5 0.5"
              dur="3s"
              repeatCount="indefinite"
            />
          )}
        </linearGradient>
      </defs>
      <path
        d="M12 1C12.6 6.6 17.4 11.4 23 12C17.4 12.6 12.6 17.4 12 23C11.4 17.4 6.6 12.6 1 12C6.6 11.4 11.4 6.6 12 1Z"
        fill={`url(#g${id})`}
      />
    </svg>
  )
}
