-- Drop the Div/Span wrappers citeproc and the JATS reader leave behind,
-- keeping their content (plain Markdown output).
return {
  { Div = function(d) return d.content end,
    Span = function(s) return s.content end },
}
