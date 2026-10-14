import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "CPU Workbench",
  description: "CPU-first game Transformer profiler and learning workspace",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable} ${mono.variable}`}>
      <body>
        <header className="app-header">
          <Link className="brand" href="/profiler">
            <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
            <span>CPU Workbench</span>
          </Link>
          <p>Measure the machine. Explain the model.</p>
          <nav aria-label="Primary navigation">
            <Link href="/profiler">Profiler</Link>
            <Link href="/learn">Learn</Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
