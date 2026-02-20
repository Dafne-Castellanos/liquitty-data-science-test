-- ============================================================================
-- Archivo:      top_10_clientes_mayor_saldo_capital.sql
-- Descripción:  Obtiene los 10 clientes con mayor saldo capital.
-- Fuente:       Tabla `clientes` (originada de data/PruebaDS.xlsx)
-- Autor:        Dafne Valeria Castellanos Rosas
-- Fecha:        2026-02-20
-- ============================================================================
--
-- CONTEXTO:
--   Esta consulta identifica a los 10 clientes que poseen el mayor saldo
--   de capital vigente. Es útil para priorizar estrategias de cobranza o
--   gestión de cartera sobre los clientes con mayor exposición financiera.
--
-- LÓGICA:
--   1. Se agrupa por cliente (identificacion) para consolidar todos sus
--      registros mensuales y obtener el saldo capital máximo histórico.
--   2. Se ordena de forma descendente por dicho saldo máximo.
--   3. Se limita el resultado a los primeros 10 registros.
--
-- NOTAS:
--   - Se utiliza MAX(saldo_capital) porque un mismo cliente puede tener
--     múltiples filas correspondientes a distintos meses de reporte.
--   - Se incluyen columnas descriptivas (tipo_documento, genero,
--     departamento, banco) tomando el valor más reciente (MAX) para dar
--     contexto al resultado.
--   - La cláusula LIMIT 10 es estándar en PostgreSQL/MySQL/SQLite.
--     En SQL Server se debe reemplazar por SELECT TOP 10.
-- ============================================================================

SELECT
    c.identificacion,
    c.tipo_documento,
    c.genero,
    c.departamento,
    c.banco,
    MAX(c.saldo_capital)    AS max_saldo_capital,
    MAX(c.dias_mora)        AS max_dias_mora
FROM
    clientes AS c
GROUP BY
    c.identificacion,
    c.tipo_documento,
    c.genero,
    c.departamento,
    c.banco
ORDER BY
    max_saldo_capital DESC
LIMIT 10;
