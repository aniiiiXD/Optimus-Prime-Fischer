import ProfilerWorkbench from "@/components/ProfilerWorkbench";
import { loadReports } from "@/lib/reports";

export default async function ProfilerPage() {
  const bundle = await loadReports();
  return <ProfilerWorkbench initialReports={bundle.reports} missing={bundle.missing} />;
}
