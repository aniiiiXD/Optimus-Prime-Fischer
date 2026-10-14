import Link from "next/link";

export default function NotFound() {
  return (
    <main className="message-page">
      <p className="context-line">404</p>
      <h1>This experiment does not exist.</h1>
      <Link href="/profiler">Return to the profiler</Link>
    </main>
  );
}
