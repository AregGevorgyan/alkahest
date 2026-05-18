import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Alkahest Demo Playground',
  description: 'Interactive demo playground for the Alkahest CAS library',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
          crossOrigin="anonymous"
        />
      </head>
      <body className="min-h-screen bg-ak-bg text-ak-fg">{children}</body>
    </html>
  );
}
