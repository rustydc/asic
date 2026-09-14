module Main (main) where

import Clash.Prelude
import FixedLLM
import qualified Prelude as P

item :: WorkItem
item = WorkItem
  { contextId = 7
  , flags = 0
  , position = 1234
  , hidden = repeat 1
  }

main :: IO ()
main = do
  let outputs = sampleN 5 (withClockResetEnable clockGen resetGen enableGen
        (asicShard (fromList [Just item])))
  case outputs P.!! 4 of
    Just result
      | contextId result == contextId item
      , position result == position item -> pure ()
    _ -> error "four-stage shard did not preserve work-item metadata"
