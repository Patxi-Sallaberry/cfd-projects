import FlowBackground from './components/FlowBackground'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import LiveDemo from './components/LiveDemo'
import Method from './components/Method'

export default function App() {
  return (
    <div id="top" className="relative">
      <FlowBackground />
      <div className="relative z-10 flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1">
          <Hero />
          <LiveDemo />
          <Method />
        </main>
        <footer className="px-4 sm:px-6 lg:px-10 py-8 text-center text-white/40 text-xs">
          Patxi Sallaberry · INSA Toulouse · CFD × PIML — built with PyTorch, exported to run in your browser.
        </footer>
      </div>
    </div>
  )
}
