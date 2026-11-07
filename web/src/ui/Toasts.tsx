import { useToasts } from './toast'
import './toasts.css'

export function Toasts() {
  const list = useToasts()
  return (
    <div className="toasts">
      {list.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`}>
          {t.text}
        </div>
      ))}
    </div>
  )
}
