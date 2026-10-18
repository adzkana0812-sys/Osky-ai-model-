import React from 'react';
import { NeuralNetworkChat } from './components/NeuralNetworkChat';
import { Brain } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-white text-black flex flex-col font-sans selection:bg-black selection:text-white">
      {/* Header - Simple, clean, flat with white background and black border */}
      <header className="bg-white border-b-2 border-black">
        <div className="max-w-4xl w-full mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-black text-white flex items-center justify-center font-black text-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-black text-black tracking-tight uppercase">
                  OSKY AI
                </h1>
                <span className="text-[10px] font-bold px-2.5 py-0.5 rounded-full border border-black bg-white text-black">
                  Neural Network Q&A
                </span>
              </div>
              <p className="text-xs text-neutral-600 font-medium">
                Tanya Jawab Jaringan Syaraf Tiruan • Multi-Bahasa & Bebas Template
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 px-3 py-1 rounded-lg border border-black text-xs font-bold bg-white">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="hidden sm:inline">Engine Aktif</span>
          </div>
        </div>
      </header>

      {/* Main Container - Dedicated Neural Network Chat View */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-6 flex flex-col justify-center">
        <NeuralNetworkChat />
      </main>

      {/* Footer - Minimalist, single-layer divider */}
      <footer className="border-t-2 border-black bg-white py-3 mt-auto">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-neutral-600 font-medium">
          <div className="flex items-center gap-2">
            <span className="font-bold text-black">OSKY AI</span>
            <span>— Antarmuka Tanya Jawab Jaringan Syaraf Tiruan</span>
          </div>
          <div className="font-mono text-black font-semibold text-right">
            Model: Deep Neural Network Engine
          </div>
        </div>
      </footer>
    </div>
  );
}
