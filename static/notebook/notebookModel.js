export function toIpynb(cells) {
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { name: 'python3', display_name: 'Python 3' },
      language_info: { name: 'python' }
    },
    cells: cells.map((c) => ({
      cell_type: c.kind,
      source: c.source.split(/(?<=\n)/),
      metadata: {},
      ...(c.kind === 'code' ? { outputs: c.outputs ?? [], execution_count: c.execution_count ?? null } : {})
    }))
  };
}

export function fromIpynb(nb) {
  return (nb.cells || []).map((c) => ({
    id: crypto.randomUUID(),
    kind: c.cell_type === 'markdown' ? 'markdown' : 'code',
    source: Array.isArray(c.source) ? c.source.join('') : (c.source || ''),
    outputs: c.outputs || [],
    execution_count: c.execution_count ?? null
  }));
}


