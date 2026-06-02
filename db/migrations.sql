-- db/migrations.sql
-- Run this ONCE manually in the Supabase SQL editor
-- (Dashboard → SQL Editor → New query → paste → Run).

CREATE TABLE signals (
  id bigserial PRIMARY KEY,
  date date NOT NULL UNIQUE,
  price numeric(10,2),
  bias text CHECK (bias IN ('BULLISH','BEARISH','NEUTRAL')),
  confidence numeric(5,2),
  signal_score numeric(8,4),
  position_size numeric(4,2),
  vol_regime text CHECK (vol_regime IN ('normal','elevated','extreme')),
  sma_200 numeric(10,2),
  circuit_breaker_active boolean DEFAULT false,
  s1_signal text, s1_driver text,
  s2_signal text, s2_driver text,
  s4_signal text, s4_driver text,
  s5_signal text, s5_driver text,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_signals_date ON signals(date DESC);


-- ── Migration: options / OI layer (run in Supabase SQL editor) ──────────────
-- Adds the options-positioning columns written by run_daily.py. Safe to re-run.
ALTER TABLE signals
  ADD COLUMN IF NOT EXISTS confidence_adjusted numeric(5,2),
  ADD COLUMN IF NOT EXISTS options_signal text,
  ADD COLUMN IF NOT EXISTS options_pcr_oi numeric(6,3),
  ADD COLUMN IF NOT EXISTS options_adjustment numeric(5,1);
