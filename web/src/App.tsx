import Dashboard from './dashboard/Dashboard'
import Editor from './editor'

// Two routes only (spec §10): "/" dashboard, "/editor" (W's route editor).
export default function App() {
  return location.pathname.startsWith('/editor') ? <Editor /> : <Dashboard />
}
