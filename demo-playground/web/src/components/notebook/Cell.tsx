'use client';

import { useCallback, useRef } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { EditorView } from '@codemirror/view';
import type { OutputItem } from '@/lib/execution';
import Output from './Output';

export interface CellData {
  id: string;
  code: string;
  outputs: OutputItem[];
  status: 'idle' | 'running' | 'done' | 'error';
  executionCount: number | null;
  backend: 'wasm' | 'server' | null;
}

interface CellProps {
  cell: CellData;
  index: number;
  onCodeChange: (id: string, code: string) => void;
  onRun: (id: string) => void;
  onDelete: (id: string) => void;
  onMoveUp: (id: string) => void;
  onMoveDown: (id: string) => void;
  onAddBelow: (id: string) => void;
}

const warmLightTheme = EditorView.theme({
  '&': {
    backgroundColor: '#eeede7',
    color: '#111',
  },
  '.cm-content': { padding: '8px 0' },
  '.cm-line': { padding: '0 12px' },
  '.cm-gutters': { backgroundColor: '#e5e4de', borderRight: '1px solid #e0ded6', color: '#888' },
  '.cm-activeLineGutter': { backgroundColor: '#dddcd6' },
  '.cm-activeLine': { backgroundColor: 'rgba(0,0,0,0.03)' },
  '.cm-selectionBackground': { backgroundColor: '#c41e3a22' },
  '&.cm-focused .cm-selectionBackground': { backgroundColor: '#c41e3a33' },
  '.cm-cursor': { borderLeftColor: '#c41e3a' },
});

export default function Cell({ cell, index, onCodeChange, onRun, onDelete, onMoveUp, onMoveDown, onAddBelow }: CellProps) {
  const editorRef = useRef<{ view?: { focus: () => void } }>(null);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        onRun(cell.id);
      }
    },
    [cell.id, onRun],
  );

  const gutter = cell.executionCount !== null
    ? `[${cell.executionCount}]`
    : cell.status === 'running' ? '[*]' : '[ ]';

  const hasError = cell.outputs.some((o) => o.type === 'error');

  return (
    <div
      className={`group relative rounded-lg border transition-all ${
        cell.status === 'running'
          ? 'border-ak-brand shadow-sm'
          : hasError
          ? 'border-red-200'
          : 'border-ak-border hover:border-ak-muted/40'
      }`}
    >
      {/* Cell header */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-ak-border bg-ak-bg rounded-t-lg">
        {/* Execution counter */}
        <span className="font-mono text-xs text-ak-muted w-8 shrink-0">{gutter}</span>

        {/* Backend badge */}
        {cell.backend && (
          <span
            className={`text-xs px-1.5 py-0.5 rounded font-mono ${
              cell.backend === 'server'
                ? 'bg-ak-brand/10 text-ak-brand'
                : 'bg-green-100 text-green-700'
            }`}
          >
            {cell.backend}
          </span>
        )}

        <div className="flex-1" />

        {/* Cell controls */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <CellBtn title="Move up" onClick={() => onMoveUp(cell.id)}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m18 15-6-6-6 6"/></svg>
          </CellBtn>
          <CellBtn title="Move down" onClick={() => onMoveDown(cell.id)}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m6 9 6 6 6-6"/></svg>
          </CellBtn>
          <CellBtn title="Add cell below" onClick={() => onAddBelow(cell.id)}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14"/></svg>
          </CellBtn>
          <CellBtn title="Delete cell" onClick={() => onDelete(cell.id)} danger>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
          </CellBtn>
        </div>

        {/* Run button */}
        <button
          onClick={() => onRun(cell.id)}
          disabled={cell.status === 'running'}
          title="Run cell (⌘ Enter)"
          className={`flex items-center gap-1 rounded px-2.5 py-1 text-xs font-medium transition-all ${
            cell.status === 'running'
              ? 'bg-ak-brand/20 text-ak-brand cursor-wait'
              : 'bg-ak-brand text-white hover:bg-ak-brand-dark'
          }`}
        >
          {cell.status === 'running' ? (
            <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          )}
          Run
        </button>
      </div>

      {/* Editor */}
      <div onKeyDown={handleKeyDown}>
        <CodeMirror
          ref={editorRef as never}
          value={cell.code}
          onChange={(code) => onCodeChange(cell.id, code)}
          extensions={[python(), warmLightTheme]}
          basicSetup={{
            lineNumbers: true,
            highlightActiveLine: true,
            highlightActiveLineGutter: true,
            autocompletion: true,
            indentOnInput: true,
            bracketMatching: true,
          }}
          style={{ fontSize: '0.875rem' }}
          className="rounded-b-lg overflow-hidden"
        />
      </div>

      {/* Output */}
      <Output items={cell.outputs} />
    </div>
  );
}

function CellBtn({
  title,
  onClick,
  danger,
  children,
}: {
  title: string;
  onClick: () => void;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`rounded p-1 transition-colors ${
        danger
          ? 'text-ak-muted hover:text-red-500 hover:bg-red-50'
          : 'text-ak-muted hover:text-ak-fg hover:bg-ak-code-bg'
      }`}
    >
      {children}
    </button>
  );
}
