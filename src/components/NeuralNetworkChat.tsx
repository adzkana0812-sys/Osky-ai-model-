import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Sparkles,
  RefreshCw,
  Cpu,
  Trash2,
  CheckCircle,
  Copy,
  Check,
  Globe,
  Brain,
  MessageSquare,
  ChevronRight,
  ShieldCheck,
} from 'lucide-react';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
  normalizedQuery?: string;
  source?: string;
}

export const NeuralNetworkChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'ai',
      text: `Halo! Saya OSKY AI Neural Network Engine.

Saya adalah sistem kecerdasan berbasis Jaringan Syaraf Tiruan (Deep Neural Network) yang siap berdiskusi, menjawab berbagai pertanyaan ilmiah, logika, matematika, pemrograman, maupun percakapan santai.

Fitur & Kemampuan Saya:
• Pemahaman Multi-Bahasa (Indonesia, English, Sunda, Jawa, dsb.)
• Toleransi Singkatan & Typo (contoh: "maksut" → "maksud", "kmu", "knp", "bisa g")
• Penalaran Murni & Bebas Template
• Penjelasan mendalam cara kerja arsitektur Neural Network (Bobot, Aktivasi, Backpropagation, dsb.)

Silakan ketik pertanyaan apa saja di bawah ini!`,
      timestamp: 'Baru saja',
      source: 'osky-neural-engine',
    },
  ]);
  const [inputValue, setInputValue] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          chatHistory: messages,
        }),
      });

      const data = await response.json();
      const reply = data.reply || 'Maaf, terjadi kendala saat memproses respons neural network.';

      const aiMsg: ChatMessage = {
        id: `ai-${Date.now()}`,
        sender: 'ai',
        text: reply,
        timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
        normalizedQuery: data.normalizedQuery,
        source: data.source || 'osky-neural-engine',
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch {
      const errorMsg: ChatMessage = {
        id: `ai-err-${Date.now()}`,
        sender: 'ai',
        text: 'Koneksi jaringan terganggu. Silakan kirim ulang pesan Anda.',
        timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearHistory = () => {
    setMessages([
      {
        id: `welcome-${Date.now()}`,
        sender: 'ai',
        text: 'Riwayat percakapan telah dibersihkan. Silakan tanyakan hal baru!',
        timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
        source: 'osky-neural-engine',
      },
    ]);
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const samplePrompts = [
    'Maksut dari backpropagation di neural network gmn?',
    'Knp fungsi aktivasi ReLU sering dipake dibanding Sigmoid?',
    'Bisa g jelaskan cara kerja transformer architecture?',
    'Bantu saya bikin algoritma binary search di Python',
    'How do artificial neural networks adjust synaptic weights? (English)',
    'Punten, tiasa ngajelaskeun perkawis kecerdasan buatan? (Sunda)',
  ];

  return (
    <div className="w-full bg-white border-2 border-black rounded-2xl flex flex-col min-h-[620px]">
      {/* Top Header Bar */}
      <div className="p-4 sm:p-5 border-b-2 border-black flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-black text-white flex items-center justify-center font-black">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-black text-black uppercase tracking-tight">
                Tanya Jawab Neural Network Engine
              </h2>
              <span className="px-2 py-0.5 rounded-full border border-black bg-white text-[10px] font-bold text-black flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                ONLINE
              </span>
            </div>
            <p className="text-xs text-neutral-600 font-medium mt-0.5">
              Inferensi AI murni multi-bahasa • Toleran typo & singkatan gaul
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-end sm:self-auto">
          <button
            type="button"
            onClick={handleClearHistory}
            className="px-3 py-1.5 rounded-lg border border-black bg-white hover:bg-neutral-100 text-black text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer"
            title="Bersihkan obrolan"
          >
            <Trash2 className="w-3.5 h-3.5 text-black" />
            <span>Hapus Chat</span>
          </button>
        </div>
      </div>

      {/* Feature Capabilities Badge Bar */}
      <div className="px-4 py-2.5 border-b border-black/20 bg-neutral-50 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] font-bold text-black flex items-center gap-1">
            <Globe className="w-3.5 h-3.5" />
            Bahasa Didukung:
          </span>
          {['Indonesia', 'English', 'Sunda', 'Jawa', 'Lainnya'].map((lang) => (
            <span
              key={lang}
              className="px-2 py-0.5 rounded border border-black bg-white text-black font-semibold text-[10px]"
            >
              {lang}
            </span>
          ))}
        </div>

        <div className="flex items-center gap-1.5 text-[11px] text-black font-semibold">
          <ShieldCheck className="w-3.5 h-3.5 text-black" />
          <span>Normalisasi Typo Aktif ("maksut" → "maksud")</span>
        </div>
      </div>

      {/* Suggested Quick Prompt Chips */}
      <div className="p-3 border-b border-black/20 bg-white">
        <div className="text-[11px] font-bold text-neutral-500 uppercase tracking-wider mb-2 flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-black" />
          <span>Contoh Topik & Pertanyaan Cepat:</span>
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
          {samplePrompts.map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSend(prompt)}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-full border border-black bg-white hover:bg-black hover:text-white text-black text-xs font-medium whitespace-nowrap transition-colors cursor-pointer shrink-0 disabled:opacity-50"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Messages Body */}
      <div className="flex-1 p-4 sm:p-6 overflow-y-auto space-y-4 max-h-[500px] min-h-[360px] bg-white">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div className="flex items-center gap-2 mb-1 px-1">
              <span className="text-[11px] font-bold font-mono text-black">
                {msg.sender === 'user' ? 'Anda' : 'OSKY AI Neural Engine'}
              </span>
              <span className="text-[10px] text-neutral-400 font-mono">
                {msg.timestamp}
              </span>
            </div>

            <div
              className={`max-w-[90%] sm:max-w-[80%] p-4 rounded-xl border-2 text-sm leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-black text-white border-black rounded-tr-none'
                  : 'bg-white text-black border-black rounded-tl-none'
              }`}
            >
              {/* Typo normalization tag */}
              {msg.normalizedQuery && (
                <div className="mb-2 pb-1.5 border-b border-black/20 text-[11px] text-neutral-700 font-medium flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-black shrink-0" />
                  <span>
                    <strong className="text-black">Typo / Singkatan Dikenali:</strong> "{msg.normalizedQuery}"
                  </span>
                </div>
              )}

              <div className="whitespace-pre-wrap">{msg.text}</div>

              {/* Message footer with copy button */}
              {msg.sender === 'ai' && (
                <div className="mt-3 pt-2 border-t border-black/10 flex items-center justify-between text-[11px] text-neutral-500 font-mono">
                  <span>Engine: Pure Neural Inference</span>
                  <button
                    type="button"
                    onClick={() => handleCopy(msg.id, msg.text)}
                    className="p-1 hover:text-black transition-colors flex items-center gap-1 cursor-pointer font-sans text-xs"
                    title="Salin jawaban"
                  >
                    {copiedId === msg.id ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                        <span className="text-emerald-600 font-bold">Tersalin</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Salin</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex flex-col items-start">
            <div className="flex items-center gap-2 mb-1 px-1">
              <span className="text-[11px] font-bold font-mono text-black">
                OSKY AI Neural Engine
              </span>
              <span className="text-[10px] text-neutral-400 font-mono">Memproses...</span>
            </div>
            <div className="p-4 rounded-xl rounded-tl-none border-2 border-black bg-neutral-50 text-black text-sm flex items-center gap-3">
              <div className="w-4 h-4 rounded-full border-2 border-black border-t-transparent animate-spin" />
              <div className="space-y-1">
                <span className="font-bold text-xs block">
                  Menjalankan Inferensi Syaraf Tiruan...
                </span>
                <span className="text-[11px] text-neutral-600 block">
                  Menganalisis token, mengekstrak representasi vektor, dan menyusun sintesis murni.
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Message Area */}
      <div className="p-4 border-t-2 border-black bg-white rounded-b-2xl">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-2"
        >
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ketik pertanyaan apa saja (misal: 'maksut neural network gmn?', konsep AI, matematika, coding)..."
              disabled={isLoading}
              className="w-full px-4 py-3.5 text-sm rounded-xl border-2 border-black bg-white text-black placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-black focus:border-black font-medium disabled:bg-neutral-100"
            />
          </div>

          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="px-5 py-3.5 rounded-xl border-2 border-black bg-black text-white hover:bg-neutral-800 font-bold text-sm flex items-center gap-2 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
          >
            <span>Kirim</span>
            <Send className="w-4 h-4" />
          </button>
        </form>

        <div className="mt-2 text-center">
          <p className="text-[11px] text-neutral-500 font-medium">
            OSKY AI memproses pertanyaan secara dinamis tanpa template kaku. Tekan <kbd className="font-mono font-bold text-black bg-neutral-100 px-1 py-0.5 rounded border border-black text-[10px]">Enter</kbd> untuk mengirim.
          </p>
        </div>
      </div>
    </div>
  );
};
