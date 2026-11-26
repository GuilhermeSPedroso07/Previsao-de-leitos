# Contexto Técnico do Projeto (RS 2024 — Bacias + SIH/SUS + CNES-SUS)

## Regras Inegociáveis de Dados (DATASUS / IBGE)
1. **SIH/SUS (`basedosdados.br_ms_sih.aihs_reduzidas`):**
   - SEMPRE deduplicar por `id_aih` (`SELECT DISTINCT`) para evitar o pico artificial de 2x em ago/2024.
   - SEMPRE agrupar por `data_internacao` (lendo competências até 2025), nunca por mês de faturamento (`ano`/`mes`).
   - Códigos municipais do SIH têm 6 dígitos; cruzar com IBGE usando `id_mun_6` (6 primeiros dígitos).
   - Filtrar RS via `SUBSTR(id_municipio_estabelecimento, 1, 2) = '43'` (evita conflito de tipo em `sigla_uf`).
   - Excluir Dengue (`A90–A92`) do regex de CIDs sensíveis a enchentes.
2. **CNES (`basedosdados.br_ms_cnes.leito`):**
   - Usar exclusivamente `quantidade_sus` (nunca somar leitos privados).
   - UTI/Complementar = `tipo_leito = '3'` (removendo zeros à esquerda).
3. **Modelagem:**
   - Estágio 1: `HydroFiLMRegionalBasinNet` (PyTorch) com roteamento causal Montante->Jusante (`shift(1)`), Ramo Monotônico Estrutural (`F.softplus`) e normalização estrita em `2010–2023`.
   - Estágio 2: `Ridge` treinado em `2011–2019` e `2023` (sem 2020–2022), ancorado na base de 2023 (sem `lag_1m` real de 2024).
   - Coortes de Urgência = formulação multiplicativa; Coortes de CIDs esparsas = formulação aditiva por 100k hab.
   - Ocupação segura alvo = 85% (`diarias / (dias_mes * 0.85)`).

## Métricas Atuais (Baseline 2024 Out-of-Time)
- Urgência Total SUS: MAE 159.26 | R² 0.9719
- Urgência UTI SUS: MAE 20.09 | R² 0.9605
- CIDs Enchente Total SUS: MAE 2.49 | R² 0.7283
- CIDs Enchente UTI SUS: MAE 0.29 | R² 0.4107 
