-- ============================================================================
-- Archivo:      promedio_clientes_con_pago_por_departamento.sql
-- Descripción:  Calcula el promedio de clientes que realizaron un pago,
--               agrupado por departamento.
-- Fuente:       Tabla `clientes` (originada de data/PruebaDS.xlsx)
-- Autor:        Dafne Valeria Castellanos Rosas
-- Fecha:        2026-02-20
-- ============================================================================
--
-- CONTEXTO:
--   Esta consulta permite conocer, para cada departamento, cuántos clientes
--   en promedio realizan un pago por mes. Es un indicador clave para evaluar
--   la efectividad de la gestión de cobranza por región geográfica.
--
-- LÓGICA:
--   1. Subconsulta (pagos_por_mes_depto): Para cada combinación de mes y
--      departamento, se cuenta el número de clientes distintos que
--      realizaron un pago (pago = 1).
--   2. Consulta externa: Se calcula el promedio (AVG) de esos conteos
--      mensuales por departamento, redondeado a 2 decimales.
--   3. Se ordena de mayor a menor promedio para facilitar el análisis.
--
-- NOTAS:
--   - Se usa COUNT(DISTINCT identificacion) para evitar contar duplicados
--     de un mismo cliente dentro del mismo mes.
--   - La columna `pago` es un indicador binario (0/1) donde 1 significa
--     que el cliente realizó un pago en ese periodo.
--   - El promedio se calcula sobre los meses disponibles en los datos,
--     lo que permite comparar departamentos de forma normalizada.
-- ============================================================================

SELECT
    pagos.departamento,
    ROUND(AVG(pagos.total_clientes_con_pago), 2)  AS promedio_clientes_con_pago,
    COUNT(pagos.mes)                                AS meses_evaluados
FROM (
    -- Subconsulta: cuenta clientes únicos con pago por mes y departamento
    SELECT
        c.mes,
        c.departamento,
        COUNT(DISTINCT c.identificacion) AS total_clientes_con_pago
    FROM
        clientes AS c
    WHERE
        c.pago = 1
    GROUP BY
        c.mes,
        c.departamento
) AS pagos
GROUP BY
    pagos.departamento
ORDER BY
    promedio_clientes_con_pago DESC;
