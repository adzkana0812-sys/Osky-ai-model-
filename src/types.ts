export type ActiveTab = 'handwriting' | 'vision' | 'llm' | 'architecture';

export interface NeuralNetworkSpec {
  id: string;
  name: string;
  category: 'Language' | 'Vision' | 'OCR' | 'Multimodal' | 'Generative';
  params: number;
  formattedParams: string;
  architecture: string;
  description: string;
  status: 'Ready' | 'Training' | 'Trained';
  lossMetric: string;
  targetAccuracy: string;
}

export interface TrainingLogPoint {
  epoch: number;
  step: number;
  loss: number;
  metricValue: number; // CER for OCR, Accuracy for Vision, Perplexity for LLM
  metricName: string;
  timestamp: string;
}

export interface HandwritingSample {
  id: string;
  label: string;
  category: 'Angka' | 'Huruf' | 'Kata' | 'Matematika';
  description: string;
  svgStrokes: string[];
}

export interface VisionCategory {
  id: string;
  name: string;
  iconName: string;
  description: string;
  sampleCount: number;
  features: string[];
}

export interface VisionSample {
  id: string;
  name: string;
  categoryId: string;
  categoryName: string;
  colorScheme: string;
  svgIcon: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'osky';
  text: string;
  timestamp: string;
  confidence?: number;
  tokensProcessed?: number;
}

export interface FileItem {
  path: string;
  name: string;
  category: string;
  description: string;
  content: string;
}
