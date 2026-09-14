{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

-- | Synthesizable structural seed for one four-layer ASIC shard.
--
-- This intentionally uses a 16-element activation and a residual fixed matrix
-- as a tractable RTL/P&R experiment. It is not an implementation of the full
-- PyTorch layer. The point is to establish the packet, stage, and immutable
-- coefficient boundaries before adding recurrent and sparse-attention engines.
module FixedLLM
  ( Activation
  , WorkItem (..)
  , asicShard
  , topEntity
  ) where

import Clash.Prelude
import GHC.Generics (Generic)

type HiddenWidth = 16
type Activation = Signed 8
type Accumulator = Signed 24
type Weight = Signed 4
type WeightRow = Vec HiddenWidth Weight
type WeightMatrix = Vec HiddenWidth WeightRow

-- | The moving state at a stage boundary. Persistent recurrent and context
-- memory are deliberately not fields of this packet.
data WorkItem = WorkItem
  { contextId :: Unsigned 5
  , flags :: BitVector 8
  , position :: Unsigned 17
  , hidden :: Vec HiddenWidth Activation
  }
  deriving (BitPack, Eq, Generic, NFDataX, Show)

-- Four literal coefficient rows stand in for four independently personalized
-- fixed fabrics. 'repeat' creates wires/constants, not an addressed weight RAM.
seedR0, seedR1, seedR2, seedG :: WeightRow
seedR0 = 1 :> 0 :> (-1) :> 2 :> 1 :> 0 :> (-2) :> 1 :> 0 :> 1 :> 0 :> (-1) :> 2 :> 0 :> 1 :> (-1) :> Nil
seedR1 = 0 :> 1 :> 1 :> (-1) :> 0 :> 2 :> 0 :> (-2) :> 1 :> 0 :> (-1) :> 1 :> 0 :> 1 :> 2 :> 0 :> Nil
seedR2 = 1 :> 1 :> 0 :> 0 :> (-1) :> 1 :> 2 :> 0 :> (-2) :> 0 :> 1 :> (-1) :> 1 :> 0 :> 0 :> 1 :> Nil
seedG  = 2 :> 0 :> 1 :> (-1) :> 1 :> 0 :> 0 :> 1 :> (-2) :> 1 :> 0 :> 1 :> 0 :> (-1) :> 1 :> 0 :> Nil

weightsR0, weightsR1, weightsR2, weightsG :: WeightMatrix
weightsR0 = repeat seedR0
weightsR1 = repeat seedR1
weightsR2 = repeat seedR2
weightsG = repeat seedG

dot :: WeightRow -> Vec HiddenWidth Activation -> Accumulator
dot coefficients activations =
  fold (+) (zipWith multiply coefficients activations)
 where
  multiply coefficient activation =
    resize coefficient * resize activation

-- | Cheap placeholder nonlinearity and requantization for the physical shell.
-- Saturation makes the arithmetic contract explicit instead of relying on
-- two's-complement wraparound.
requantize :: Accumulator -> Activation
requantize value = satResize SatBound (shiftR value 3)

fixedResidual :: WeightMatrix -> Vec HiddenWidth Activation -> Vec HiddenWidth Activation
fixedResidual coefficients input = zipWith addResidual input projected
 where
  projected = map (requantize . (`dot` input)) coefficients
  addResidual old new = satResize SatBound ((resize old :: Signed 9) + resize new)

runFixedLayer :: WeightMatrix -> WorkItem -> WorkItem
runFixedLayer coefficients item@WorkItem{..} =
  item {hidden = fixedResidual coefficients hidden}

registeredLayer
  :: HiddenClockResetEnable dom
  => WeightMatrix
  -> Signal dom (Maybe WorkItem)
  -> Signal dom (Maybe WorkItem)
registeredLayer coefficients = register Nothing . fmap (fmap (runFixedLayer coefficients))

-- | Four independent physical stages in the intended R/R/R/G order. Each
-- stage accepts one item per cycle after fill. The G stage currently contains
-- only the fixed projection shell; its context-memory engine is the next RTL
-- milestone.
asicShard
  :: HiddenClockResetEnable dom
  => Signal dom (Maybe WorkItem)
  -> Signal dom (Maybe WorkItem)
asicShard input = globalStage
 where
  recurrent0 = registeredLayer weightsR0 input
  recurrent1 = registeredLayer weightsR1 recurrent0
  recurrent2 = registeredLayer weightsR2 recurrent1
  globalStage = registeredLayer weightsG recurrent2

{-# ANN topEntity
  (Synthesize
    { t_name = "fixed_llm_asic_shard"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "work_in"
        ]
    , t_output = PortName "work_out"
    }) #-}
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Maybe WorkItem)
  -> Signal System (Maybe WorkItem)
topEntity = exposeClockResetEnable asicShard

