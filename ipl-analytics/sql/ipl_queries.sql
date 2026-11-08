-- Queries below run against SQLite tables built from the cleaned data by src/run_analysis.py:
--   matches   one row per match (data/processed/matches_clean.csv)
--   innings   one row per innings (innings_summary.csv)
--   batting   batter x season x team x phase (batting_by_phase.csv)
--   bowling   bowler x season x team x phase (bowling_by_phase.csv)
-- "Regular" matches exclude no-results and rain-affected (D/L) games, whose scores are not comparable.
-- Each query is introduced by a title comment line.

-- QUERY: Do toss winners win more often? (overall and by toss decision)
SELECT toss_decision,
       COUNT(*)                                                   AS matches,
       SUM(toss_winner_won)                                       AS toss_winner_wins,
       ROUND(100.0 * SUM(toss_winner_won) / COUNT(*), 1)          AS toss_winner_win_pct
FROM matches
WHERE decided = 1
GROUP BY toss_decision
UNION ALL
SELECT 'All', COUNT(*), SUM(toss_winner_won), ROUND(100.0 * SUM(toss_winner_won) / COUNT(*), 1)
FROM matches
WHERE decided = 1;

-- QUERY: Chasing success by season with a 3-season rolling rate (window function)
WITH s AS (
    SELECT year, COUNT(*) AS matches, SUM(chase_won) AS chase_wins
    FROM matches
    WHERE chase_won IS NOT NULL
    GROUP BY year
)
SELECT year, matches, chase_wins,
       ROUND(100.0 * chase_wins / matches, 1) AS chase_win_pct,
       ROUND(100.0 * SUM(chase_wins) OVER w / SUM(matches) OVER w, 1) AS rolling_3_season_pct
FROM s
WINDOW w AS (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
ORDER BY year;

-- QUERY: Share of toss winners who choose to field, by season
SELECT year,
       COUNT(*)                                         AS matches,
       ROUND(100.0 * SUM(toss_chose_field) / COUNT(*), 1) AS chose_to_field_pct
FROM matches
GROUP BY year
ORDER BY year;

-- QUERY: Runs per over by phase and season (powerplay = overs 1-6, middle = 7-15, death = 16-20)
SELECT year,
       ROUND(6.0 * SUM(pp_runs)    / SUM(pp_balls),    2) AS powerplay_rpo,
       ROUND(6.0 * SUM(mid_runs)   / SUM(mid_balls),   2) AS middle_rpo,
       ROUND(6.0 * SUM(death_runs) / SUM(death_balls), 2) AS death_rpo
FROM innings
WHERE no_result = 0 AND is_dl = 0
GROUP BY year
ORDER BY year;

-- QUERY: Average first-innings score by season and change vs the previous season (LAG)
WITH f AS (
    SELECT year, AVG(runs) AS avg_first
    FROM innings
    WHERE inning = 1 AND no_result = 0 AND is_dl = 0
    GROUP BY year
)
SELECT year,
       ROUND(avg_first, 1)                                         AS avg_first_innings,
       ROUND(avg_first - LAG(avg_first) OVER (ORDER BY year), 1)   AS change_vs_prev_season
FROM f
ORDER BY year;

-- QUERY: 200+ totals and sixes per match by season
SELECT year,
       SUM(CASE WHEN runs >= 200 THEN 1 ELSE 0 END)                AS scores_of_200_plus,
       ROUND(1.0 * SUM(sixes) / COUNT(DISTINCT match_id), 1)       AS sixes_per_match
FROM innings
WHERE no_result = 0
GROUP BY year
ORDER BY year;

-- QUERY: Era comparison - first-innings scoring before vs after the Impact Player rule (2023)
SELECT CASE WHEN year >= 2023 THEN '2023-24 (Impact Player era)' ELSE '2008-2022' END AS era,
       COUNT(*)                                                    AS first_innings,
       ROUND(AVG(runs), 1)                                         AS avg_runs,
       ROUND(100.0 * SUM(CASE WHEN runs >= 200 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_scores_200_plus
FROM innings
WHERE inning = 1 AND no_result = 0 AND is_dl = 0
GROUP BY era
ORDER BY era;

-- QUERY: Ten highest team totals (RANK window function)
WITH ranked AS (
    SELECT RANK() OVER (ORDER BY runs DESC) AS rnk,
           batting_team, bowling_team, runs, wickets, year, venue
    FROM innings
)
SELECT rnk, batting_team, bowling_team, runs, wickets, year, venue
FROM ranked
WHERE rnk <= 10
ORDER BY rnk, year;

-- QUERY: Chase win rate by target band (what total is safe to defend?)
SELECT CASE WHEN first_runs + 1 <= 140 THEN '1) 140 or less'
            WHEN first_runs + 1 <= 160 THEN '2) 141-160'
            WHEN first_runs + 1 <= 180 THEN '3) 161-180'
            WHEN first_runs + 1 <= 200 THEN '4) 181-200'
            ELSE '5) 201 or more' END                              AS target_band,
       COUNT(*)                                                    AS matches,
       SUM(chase_won)                                              AS chases_won,
       ROUND(100.0 * SUM(chase_won) / COUNT(*), 1)                 AS chase_win_pct
FROM matches
WHERE chase_won IS NOT NULL AND is_dl = 0
GROUP BY target_band
ORDER BY target_band;

-- QUERY: Powerplay runs and winning (both innings, regular matches)
SELECT inning,
       CASE WHEN pp_runs < 40 THEN '1) under 40'
            WHEN pp_runs < 50 THEN '2) 40-49'
            WHEN pp_runs < 60 THEN '3) 50-59'
            ELSE '4) 60 or more' END                               AS powerplay_runs,
       COUNT(*)                                                    AS innings,
       ROUND(100.0 * SUM(won) / COUNT(*), 1)                       AS batting_team_win_pct
FROM innings
WHERE won IS NOT NULL AND is_dl = 0
GROUP BY inning, powerplay_runs
ORDER BY inning, powerplay_runs;

-- QUERY: Venue guide - scoring and chasing at grounds with 20+ regular matches (RANK)
SELECT RANK() OVER (ORDER BY AVG(first_runs) DESC)                 AS scoring_rank,
       venue,
       COUNT(*)                                                    AS matches,
       ROUND(AVG(first_runs), 1)                                   AS avg_first_innings,
       ROUND(100.0 * SUM(chase_won) / COUNT(chase_won), 1)         AS chase_win_pct
FROM matches
WHERE no_result = 0 AND is_dl = 0
GROUP BY venue
HAVING COUNT(*) >= 20
ORDER BY scoring_rank;

-- QUERY: League-wide home advantage (true home-and-away seasons only)
SELECT COUNT(*)                                                    AS matches_with_a_home_team,
       SUM(CASE WHEN winner = home_team THEN 1 ELSE 0 END)         AS home_wins,
       ROUND(100.0 * SUM(CASE WHEN winner = home_team THEN 1 ELSE 0 END) / COUNT(*), 1) AS home_win_pct
FROM matches
WHERE home_team IS NOT NULL AND decided = 1;

-- QUERY: Home vs away win rate by franchise (CTE + UNION ALL, at least 15 home and 15 away matches)
WITH team_matches AS (
    SELECT home_team AS team, 'Home' AS side, CASE WHEN winner = home_team THEN 1 ELSE 0 END AS won
    FROM matches WHERE home_team IS NOT NULL AND decided = 1
    UNION ALL
    SELECT away_team, 'Away', CASE WHEN winner = away_team THEN 1 ELSE 0 END
    FROM matches WHERE away_team IS NOT NULL AND decided = 1
),
agg AS (
    SELECT team,
           SUM(CASE WHEN side = 'Home' THEN 1 ELSE 0 END)          AS home_matches,
           ROUND(100.0 * SUM(CASE WHEN side = 'Home' THEN won END) / SUM(CASE WHEN side = 'Home' THEN 1 ELSE 0 END), 1) AS home_win_pct,
           SUM(CASE WHEN side = 'Away' THEN 1 ELSE 0 END)          AS away_matches,
           ROUND(100.0 * SUM(CASE WHEN side = 'Away' THEN won END) / SUM(CASE WHEN side = 'Away' THEN 1 ELSE 0 END), 1) AS away_win_pct
    FROM team_matches
    GROUP BY team
)
SELECT RANK() OVER (ORDER BY home_win_pct - away_win_pct DESC)     AS rank_by_home_boost,
       team, home_matches, home_win_pct, away_matches, away_win_pct,
       ROUND(home_win_pct - away_win_pct, 1)                       AS home_boost_pts
FROM agg
WHERE home_matches >= 15 AND away_matches >= 15
ORDER BY rank_by_home_boost;

-- QUERY: Hyderabad franchises - results at Rajiv Gandhi Stadium (Uppal) vs everywhere else
WITH t AS (
    SELECT team1 AS team, venue, winner, decided FROM matches
    UNION ALL
    SELECT team2, venue, winner, decided FROM matches
)
SELECT team,
       CASE WHEN venue = 'Rajiv Gandhi International Stadium, Hyderabad' THEN 'At Hyderabad' ELSE 'Elsewhere' END AS where_played,
       COUNT(*)                                                    AS matches,
       SUM(CASE WHEN winner = team THEN 1 ELSE 0 END)              AS wins,
       ROUND(100.0 * SUM(CASE WHEN winner = team THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_pct
FROM t
WHERE decided = 1 AND team IN ('Sunrisers Hyderabad', 'Deccan Chargers')
GROUP BY team, where_played
ORDER BY team, where_played;

-- QUERY: Uppal as a ground - scoring and chasing vs the rest of the league
SELECT CASE WHEN venue = 'Rajiv Gandhi International Stadium, Hyderabad' THEN 'Uppal, Hyderabad' ELSE 'All other grounds' END AS ground,
       COUNT(*)                                                    AS matches,
       ROUND(AVG(first_runs), 1)                                   AS avg_first_innings,
       ROUND(100.0 * SUM(chase_won) / COUNT(chase_won), 1)         AS chase_win_pct,
       ROUND(100.0 * SUM(toss_chose_field) / COUNT(*), 1)          AS chose_to_field_pct
FROM matches
WHERE no_result = 0 AND is_dl = 0
GROUP BY ground
ORDER BY ground DESC;

-- QUERY: Franchise table - titles, matches and win rate (RANK, franchises with 40+ matches)
WITH titles AS (
    SELECT winner AS team, COUNT(*) AS titles FROM matches WHERE match_type = 'Final' GROUP BY winner
),
rec AS (
    SELECT team, COUNT(*) AS matches, SUM(won) AS wins
    FROM (
        SELECT team1 AS team, CASE WHEN winner = team1 THEN 1 ELSE 0 END AS won FROM matches WHERE decided = 1
        UNION ALL
        SELECT team2, CASE WHEN winner = team2 THEN 1 ELSE 0 END FROM matches WHERE decided = 1
    )
    GROUP BY team
)
SELECT RANK() OVER (ORDER BY COALESCE(t.titles, 0) DESC, 1.0 * r.wins / r.matches DESC) AS rnk,
       r.team, COALESCE(t.titles, 0) AS titles, r.matches, r.wins,
       ROUND(100.0 * r.wins / r.matches, 1) AS win_pct
FROM rec r
LEFT JOIN titles t ON t.team = r.team
WHERE r.matches >= 40
ORDER BY rnk;

-- QUERY: Top run scorers of all time with strike rate and average
SELECT RANK() OVER (ORDER BY SUM(runs) DESC)                       AS rnk,
       player,
       SUM(runs)                                                   AS runs,
       SUM(balls)                                                  AS balls,
       ROUND(100.0 * SUM(runs) / SUM(balls), 1)                    AS strike_rate,
       ROUND(1.0 * SUM(runs) / NULLIF(SUM(dismissals), 0), 1)      AS average,
       SUM(sixes)                                                  AS sixes
FROM batting
GROUP BY player
ORDER BY runs DESC
LIMIT 10;

-- QUERY: Best death-over bowlers by economy (overs 16-20, at least 300 balls)
SELECT RANK() OVER (ORDER BY 6.0 * SUM(runs) / SUM(balls))          AS rnk,
       player,
       SUM(balls)                                                  AS balls,
       ROUND(6.0 * SUM(runs) / SUM(balls), 2)                      AS economy,
       SUM(wickets)                                                AS wickets,
       ROUND(100.0 * SUM(dots) / SUM(balls), 1)                    AS dot_ball_pct
FROM bowling
WHERE phase = 'Death'
GROUP BY player
HAVING SUM(balls) >= 300
ORDER BY rnk
LIMIT 10;

-- QUERY: Fastest powerplay batters by strike rate (overs 1-6, at least 600 balls)
SELECT RANK() OVER (ORDER BY 100.0 * SUM(runs) / SUM(balls) DESC)   AS rnk,
       player,
       SUM(balls)                                                  AS balls,
       SUM(runs)                                                   AS runs,
       ROUND(100.0 * SUM(runs) / SUM(balls), 1)                    AS strike_rate,
       ROUND(100.0 * (SUM(fours) + SUM(sixes)) / SUM(balls), 1)    AS boundary_ball_pct
FROM batting
WHERE phase = 'Powerplay'
GROUP BY player
HAVING SUM(balls) >= 600
ORDER BY rnk
LIMIT 10;

-- QUERY: Player of the match leaders
SELECT RANK() OVER (ORDER BY COUNT(*) DESC)                        AS rnk,
       player_of_match                                             AS player,
       COUNT(*)                                                    AS awards
FROM matches
WHERE player_of_match IS NOT NULL
GROUP BY player_of_match
ORDER BY awards DESC, player
LIMIT 10;
