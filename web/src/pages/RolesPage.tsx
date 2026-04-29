import { useEffect, useState, useCallback } from "react";
import {
  Users,
  Trophy,
  Star,
  Target,
  Shield,
  TrendingUp,
  Award,
  RefreshCw,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  RolesResponse,
  KpiResponse,
  XpResponse,
  AchievementsResponse,
  LeaderboardResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/i18n";

function SummaryCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{label}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </CardContent>
    </Card>
  );
}

function formatPercent(n: number | null): string {
  if (n === null || n === undefined) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

function formatNumber(n: number | null): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(Math.round(n));
}

function formatDate(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return String(ts);
  }
}

export default function RolesPage() {
  const [rolesData, setRolesData] = useState<RolesResponse | null>(null);
  const [kpiData, setKpiData] = useState<KpiResponse | null>(null);
  const [xpData, setXpData] = useState<XpResponse | null>(null);
  const [achievementsData, setAchievementsData] =
    useState<AchievementsResponse | null>(null);
  const [leaderboardData, setLeaderboardData] =
    useState<LeaderboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.getRoles(),
      api.getKpi(undefined, 7),
      api.getXp(),
      api.getAchievements(),
      api.getLeaderboard(10),
    ])
      .then(([roles, kpi, xp, achievements, leaderboard]) => {
        setRolesData(roles);
        setKpiData(kpi);
        setXpData(xp);
        setAchievementsData(achievements);
        setLeaderboardData(leaderboard);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, [load]);

  const currentRole =
    rolesData?.roles.find((r) => r.name === "default") ?? rolesData?.roles[0];

  const hasAnyData =
    (rolesData && rolesData.roles.length > 0) ||
    (achievementsData && achievementsData.achievements.length > 0) ||
    (leaderboardData && leaderboardData.leaderboard.length > 0);

  return (
    <div className="flex flex-col gap-6">
      {/* Header + manual refresh */}
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold tracking-tight">{t.roles.title}</h1>
        <Button variant="outline" size="sm" className="text-xs h-7" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
          {t.common.refresh}
        </Button>
      </div>

      {loading && !hasAnyData && (
        <div className="flex items-center justify-center py-24">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        </div>
      )}

      {error && (
        <Card>
          <CardContent className="py-6">
            <p className="text-sm text-destructive text-center">{error}</p>
          </CardContent>
        </Card>
      )}

      {hasAnyData && (
        <>
          {/* Summary cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <SummaryCard
              icon={Users}
              label={t.roles.currentRole}
              value={currentRole?.name ?? t.common.none}
              sub={
                currentRole
                  ? `${currentRole.resolved_tool_count} ${t.common.tools}`
                  : undefined
              }
            />
            <SummaryCard
              icon={Star}
              label={`${t.roles.level} / ${t.roles.xp}`}
              value={xpData ? `${xpData.level}` : "—"}
              sub={
                xpData
                  ? `${formatNumber(xpData.xp)} / ${formatNumber(xpData.xp_to_next)} ${t.roles.xpToNext}`
                  : undefined
              }
            />
            <SummaryCard
              icon={Target}
              label={t.roles.kpi}
              value={formatPercent(kpiData?.task_success_rate)}
              sub={
                kpiData
                  ? `${formatNumber(kpiData.avg_tokens_per_task)} ${t.roles.avgTokens}`
                  : undefined
              }
            />
          </div>

          {/* KPI detail cards */}
          {kpiData && kpiData.record_count > 0 && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">
                    {t.roles.successRate}
                  </CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {formatPercent(kpiData.task_success_rate)}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">
                    {t.roles.avgTokens}
                  </CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {formatNumber(kpiData.avg_tokens_per_task)}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">
                    {t.roles.toolDiversity}
                  </CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {formatPercent(kpiData.tool_diversity_score)}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">
                    {t.roles.records}
                  </CardTitle>
                  <Award className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {formatNumber(kpiData.record_count)}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* All Roles table */}
          {rolesData && rolesData.roles.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">{t.roles.allRoles}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground text-xs">
                        <th className="text-left py-2 pr-4 font-medium">
                          {t.roles.currentRole}
                        </th>
                        <th className="text-left py-2 px-4 font-medium">
                          {t.common.tools}
                        </th>
                        <th className="text-left py-2 pl-4 font-medium">
                          KPI Weights
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {rolesData.roles.map((role) => (
                        <tr
                          key={role.name}
                          className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
                        >
                          <td className="py-2 pr-4">
                            <div className="font-medium">{role.name}</div>
                            {role.description && (
                              <div className="text-xs text-muted-foreground">
                                {role.description}
                              </div>
                            )}
                          </td>
                          <td className="text-left py-2 px-4 text-muted-foreground">
                            {role.resolved_tool_count}
                          </td>
                          <td className="text-left py-2 pl-4 text-muted-foreground">
                            {Object.entries(role.kpi_weights)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(", ") || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Achievements table */}
          {achievementsData && achievementsData.achievements.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Trophy className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">{t.roles.achievements}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground text-xs">
                        <th className="text-left py-2 pr-4 font-medium">
                          {t.roles.name}
                        </th>
                        <th className="text-left py-2 px-4 font-medium">
                          {t.roles.currentRole}
                        </th>
                        <th className="text-right py-2 pl-4 font-medium">
                          {t.roles.leaderboard}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {achievementsData.achievements.map((a) => (
                        <tr
                          key={a.achievement_id}
                          className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
                        >
                          <td className="py-2 pr-4">
                            <div className="font-medium">{a.name}</div>
                            {a.description && (
                              <div className="text-xs text-muted-foreground">
                                {a.description}
                              </div>
                            )}
                          </td>
                          <td className="text-left py-2 px-4 text-muted-foreground">
                            {a.role ?? "—"}
                          </td>
                          <td className="text-right py-2 pl-4 text-muted-foreground">
                            {formatDate(a.unlocked_at)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Leaderboard table */}
          {leaderboardData && leaderboardData.leaderboard.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Trophy className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">{t.roles.leaderboard}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground text-xs">
                        <th className="text-left py-2 pr-4 font-medium">
                          {t.roles.rank}
                        </th>
                        <th className="text-left py-2 px-4 font-medium">
                          {t.roles.skill}
                        </th>
                        <th className="text-right py-2 px-4 font-medium">
                          {t.roles.level}
                        </th>
                        <th className="text-right py-2 pl-4 font-medium">
                          {t.roles.xp}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {leaderboardData.leaderboard.map((entry) => (
                        <tr
                          key={`${entry.rank}-${entry.skill_name}`}
                          className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
                        >
                          <td className="py-2 pr-4 font-medium">
                            #{entry.rank}
                          </td>
                          <td className="text-left py-2 px-4">
                            {entry.skill_name}
                          </td>
                          <td className="text-right py-2 px-4">
                            {entry.level}
                          </td>
                          <td className="text-right py-2 pl-4 text-muted-foreground">
                            {formatNumber(entry.xp)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {!loading && !hasAnyData && (
        <Card>
          <CardContent className="py-12">
            <div className="flex flex-col items-center text-muted-foreground">
              <Users className="h-8 w-8 mb-3 opacity-40" />
              <p className="text-sm font-medium">{t.roles.noData}</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
