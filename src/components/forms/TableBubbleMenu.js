import React from 'react';
import { BubbleMenu } from '@tiptap/react/menus';
import {
  Table as TableIcon,
  Plus,
  Minus,
  Trash2,
  Merge,
  Split,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
} from 'lucide-react';

const TableBubbleMenu = ({ editor }) => {
  if (!editor) return null;

  const ButtonGroup = ({ children, label }) => (
    <div className="flex flex-col gap-1">
      {label && <span className="text-xs text-gray-500 px-2">{label}</span>}
      <div className="flex gap-1">{children}</div>
    </div>
  );

  const MenuButton = ({ onClick, disabled, children, title }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`p-2 rounded hover:bg-gray-100 transition-colors ${
        disabled ? 'opacity-30 cursor-not-allowed' : 'text-gray-700'
      }`}
      type="button"
    >
      {children}
    </button>
  );

  const Divider = () => <div className="w-px h-8 bg-gray-300 mx-1" />;

  return (
    <BubbleMenu
      editor={editor}
      options={{
        offset: 8,
        placement: 'top',
      }}
      shouldShow={({ editor, view, state, from, to }) => {
        // Show only when inside a table cell
        const isInTableCell = editor.isActive('tableCell') || editor.isActive('tableHeader');
        return isInTableCell;
      }}
      className="bubble-menu-table bg-white border border-gray-300 rounded-lg shadow-lg p-2 flex gap-2 items-start"
    >
      {/* Row Operations */}
      <ButtonGroup label="Righe">
        <MenuButton
          onClick={() => editor.chain().focus().addRowBefore().run()}
          title="Aggiungi riga sopra"
        >
          <div className="flex flex-col items-center">
            <ArrowUp size={14} />
            <Plus size={12} />
          </div>
        </MenuButton>
        <MenuButton
          onClick={() => editor.chain().focus().addRowAfter().run()}
          title="Aggiungi riga sotto"
        >
          <div className="flex flex-col items-center">
            <Plus size={12} />
            <ArrowDown size={14} />
          </div>
        </MenuButton>
        <MenuButton
          onClick={() => editor.chain().focus().deleteRow().run()}
          title="Elimina riga"
        >
          <Minus size={16} />
        </MenuButton>
      </ButtonGroup>

      <Divider />

      {/* Column Operations */}
      <ButtonGroup label="Colonne">
        <MenuButton
          onClick={() => editor.chain().focus().addColumnBefore().run()}
          title="Aggiungi colonna a sinistra"
        >
          <div className="flex items-center">
            <ArrowLeft size={14} />
            <Plus size={12} />
          </div>
        </MenuButton>
        <MenuButton
          onClick={() => editor.chain().focus().addColumnAfter().run()}
          title="Aggiungi colonna a destra"
        >
          <div className="flex items-center">
            <Plus size={12} />
            <ArrowRight size={14} />
          </div>
        </MenuButton>
        <MenuButton
          onClick={() => editor.chain().focus().deleteColumn().run()}
          title="Elimina colonna"
        >
          <Minus size={16} />
        </MenuButton>
      </ButtonGroup>

      <Divider />

      {/* Cell Operations */}
      <ButtonGroup label="Celle">
        <MenuButton
          onClick={() => editor.chain().focus().mergeCells().run()}
          disabled={!editor.can().mergeCells()}
          title="Unisci celle"
        >
          <Merge size={16} />
        </MenuButton>
        <MenuButton
          onClick={() => editor.chain().focus().splitCell().run()}
          disabled={!editor.can().splitCell()}
          title="Dividi cella"
        >
          <Split size={16} />
        </MenuButton>
      </ButtonGroup>

      <Divider />

      {/* Table Operations */}
      <ButtonGroup label="Tabella">
        <MenuButton
          onClick={() => editor.chain().focus().deleteTable().run()}
          title="Elimina tabella"
        >
          <Trash2 size={16} className="text-red-600" />
        </MenuButton>
      </ButtonGroup>

      <style jsx>{`
        .bubble-menu-table {
          animation: fadeIn 0.15s ease-in-out;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-5px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </BubbleMenu>
  );
};

export default TableBubbleMenu;
