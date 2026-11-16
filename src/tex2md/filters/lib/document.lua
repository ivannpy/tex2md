-- Whole-document steps: figures reduced to their caption, abstract as first section, run statistics.
local S = require("state")
local M = {}

-- Figures keep only their caption (floats.lua prints it); drop the images themselves.
function M.Image(_) return {} end

function M.Pandoc(doc)
  local blocks = pandoc.Blocks({})
  if doc.meta.abstract then
    blocks:insert(pandoc.Header(2, "Abstract", pandoc.Attr("", { "unnumbered" })))
    local abs = doc.meta.abstract
    if pandoc.utils.type(abs) == "Blocks" then
      blocks:extend(abs)
    else
      blocks:insert(pandoc.Para(abs))
    end
  end
  blocks:extend(doc.blocks)
  doc.blocks = blocks
  io.stderr:write(string.format("tex2md-stats figures=%d tables=%d\n", S.stats.figures, S.stats.tables))
  return doc
end

return M
