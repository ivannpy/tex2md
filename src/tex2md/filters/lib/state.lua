-- Shared mutable state of one pandoc run. Every module reads and writes through this table,
-- so the pass order in tex2md.lua is the only coupling between them.
local S = {
  labels = {},                      -- .aux \newlabel table: id -> {num, type}
  floats = {},                      -- .aux \caption lines in document order: {kind, num, text}
  fptr = { figure = 1, table = 1 }, -- where to start looking in S.floats, per kind
  used = { figure = {}, table = {} }, -- entries of S.floats already given to a float (the .aux lists floats in
                                    -- placement order, which is not always document order)
  flast = {},                       -- last matched float per kind (to detect multi-panel floats)
  sections = {},                    -- .aux table of contents in order: {level, num, text}
  sptr = 1,                         -- first entry of S.sections not yet given to a header
  level_shift = nil,                -- header level minus toc level (2 when a \part pushed every section down)
  counters = { 0, 0, 0 },           -- running section / subsection / subsubsection numbers
  in_appendix = false,
  fig_count = 0,                    -- fallback numbering when the .aux has no float data
  tab_count = 0,
  stats = { figures = 0, tables = 0 },
}

-- Load a JSON file named by a metadata field (-M key=path); missing or empty -> nil.
function S.read_json(meta, key)
  local path = meta[key] and pandoc.utils.stringify(meta[key])
  if not path or path == "" then return nil end
  local fh = io.open(path, "r")
  if not fh then return nil end
  local data = pandoc.json.decode(fh:read("a"))
  fh:close()
  return data
end

return S
