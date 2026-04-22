import type {
  KpiSummaryResponse,
  RoleListResponse,
  XpStatusResponse,
  AchievementsListResponse,
  LeaderboardResponse
} from '../../../gatewayTypes.js'
import type { PanelSection } from '../../../types.js'
import type { SlashCommand } from '../types.js'

export const roleCommands: SlashCommand[] = [
  {
    help: 'show role list, KPI metrics, XP/level, and achievements',
    name: 'roles',
    run: (_arg, ctx) => {
      const { rpc } = ctx.gateway
      const { panel, sys } = ctx.transcript

      Promise.all([
        rpc<RoleListResponse>('role.list', {}),
        rpc<KpiSummaryResponse>('kpi.summary', {}),
        rpc<XpStatusResponse>('xp.status', {}),
        rpc<AchievementsListResponse>('achievements.list', {}),
        rpc<LeaderboardResponse>('leaderboard', { limit: 5 })
      ])
        .then(
          ctx.guarded<[
            RoleListResponse | null,
            KpiSummaryResponse | null,
            XpStatusResponse | null,
            AchievementsListResponse | null,
            LeaderboardResponse | null
          ]>(([roleRes, kpiRes, xpRes, achRes, lbRes]) => {
            const sections: PanelSection[] = []

            // Section 1: Role list
            if (roleRes?.roles?.length) {
              sections.push({
                items: roleRes.roles.map(
                  r => `${r.name} — ${r.description ?? ''} (${r.resolved_tool_count ?? 0} tools)`
                ),
                title: 'Available Roles'
              })
            }

            // Section 2: KPI metrics (if summary has data)
            if (kpiRes?.record_count && kpiRes.record_count > 0) {
              sections.push({
                rows: [
                  ['Records', String(kpiRes.record_count)],
                  ['Success Rate', `${((kpiRes.task_success_rate ?? 0) * 100).toFixed(1)}%`],
                  ['Avg Tokens/Task', String(Math.round(kpiRes.avg_tokens_per_task ?? 0))],
                  ['Tool Diversity', `${((kpiRes.tool_diversity_score ?? 0) * 100).toFixed(1)}%`],
                  ['Error Recovery', `${((kpiRes.error_recovery_rate ?? 0) * 100).toFixed(1)}%`],
                  ['Proficiency', `${((kpiRes.role_proficiency_score ?? 0) * 100).toFixed(1)}%`]
                ],
                title: `KPI — ${kpiRes.role ?? 'all roles'}`
              })
            }

            // Section 3: XP/Level
            if (xpRes?.skill_name) {
              sections.push({
                rows: [
                  ['Level', String(xpRes.level)],
                  ['XP', `${Math.round(xpRes.xp)} / ${Math.round(xpRes.xp + xpRes.xp_to_next)}`],
                  ['To Next Level', String(Math.round(xpRes.xp_to_next))]
                ],
                title: `Progress — ${xpRes.skill_name}`
              })
            }

            // Section 4: Achievements
            if (achRes?.achievements?.length) {
              sections.push({
                items: achRes.achievements.slice(0, 5).map(a => `${a.name}${a.role ? ` (${a.role})` : ''}`),
                title: 'Achievements'
              })
            }

            // Section 5: Leaderboard
            if (lbRes?.leaderboard?.length) {
              sections.push({
                rows: lbRes.leaderboard.map((e, i) => [
                  `#${i + 1} ${e.skill_name}`,
                  `Lv${e.level} · ${Math.round(e.xp)} XP`
                ]),
                title: 'Top Roles'
              })
            }

            if (!sections.length) {
              return sys('no role or KPI data available')
            }

            panel('Roles & KPI Dashboard', sections)
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
