-- tex2md pandoc filter (entry point).
--
-- Numbers headers, figures, tables and equations as the compiled PDF does. Inputs, passed with
-- `-M key=path` (JSON files written by tex2md):
--   tex2md_labels  the .aux \newlabel table        (sections, equations, float labels)
--   tex2md_floats  the .aux \caption lines in order (the numbers of figures and tables)
--
-- Modules live in lib/; the passes below run in order and the order matters (headers are numbered
-- before floats, floats before images are dropped).
local dir = (PANDOC_SCRIPT_FILE or ""):match("^(.*)/[^/]*$") or "."
package.path = dir .. "/lib/?.lua;" .. package.path

local S = require("state")
local headers = require("headers")
local floats = require("floats")
local equations = require("equations")
local document = require("document")
local cleanup = require("cleanup")

return {
  { Meta = function(meta)
      S.labels = S.read_json(meta, "tex2md_labels") or {}
      S.floats = S.read_json(meta, "tex2md_floats") or {}
      return meta
    end },
  { Para = headers.Para, Header = headers.Header },
  { Figure = floats.Figure },
  { Table = floats.Table, Div = floats.DeferredTable, Math = equations.Math, Image = document.Image },
  { Table = cleanup.Table },
  { Div = cleanup.Div, Span = cleanup.Span, SmallCaps = cleanup.SmallCaps },
  { Pandoc = document.Pandoc },
}
