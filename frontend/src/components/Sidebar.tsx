import React from "react";
import {
  MessageSquarePlus,
  Search,
  Library,
  LayoutGrid,
  Code2,
  MoreHorizontal,
  FolderOpen,
  MessageSquare,
  User,
  PanelLeftClose
} from "lucide-react";

const Sidebar: React.FC = () => {
  return (
    <div className="w-64 h-full bg-gpt-sidebar text-gray-200 flex flex-col transition-all duration-300">
      {/* Top Header */}
      <div className="p-3 flex justify-between items-center">
        <button className="flex items-center gap-2 hover:bg-gray-800 p-2 rounded-md transition-colors flex-1">
          <div className="bg-white p-1 rounded-full text-black">
             {/* ChatGPT Icon placeholder */}
             <Code2 size={16} />
          </div>
          <span className="font-semibold text-sm">New chat</span>
          <MessageSquarePlus size={16} className="ml-auto text-gray-400" />
        </button>
        <button className="ml-2 hover:bg-gray-800 p-2 rounded-md transition-colors text-gray-400">
          <PanelLeftClose size={20} />
        </button>
      </div>

      {/* Main Navigation */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-6 text-sm">
        <div className="space-y-1">
          <NavItem icon={<Search size={18} />} label="Search chats" />
          <NavItem icon={<Library size={18} />} label="Library" />
          <NavItem icon={<LayoutGrid size={18} />} label="Apps" />
          <NavItem icon={<Code2 size={18} />} label="Codex" />
          <NavItem icon={<MoreHorizontal size={18} />} label="More" />
        </div>

        <div className="space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 px-2 mb-2">Projects</h3>
          <NavItem icon={<FolderOpen size={18} />} label="New project" />
          <NavItem icon={<FolderOpen size={18} />} label="Sanjeevani" />
          <NavItem icon={<FolderOpen size={18} />} label="github" />
          <NavItem icon={<FolderOpen size={18} />} label="Linkedin" />
          <NavItem icon={<FolderOpen size={18} />} label="vani" />
        </div>

        <div className="space-y-1">
          <h3 className="text-xs font-semibold text-gray-500 px-2 mb-2">Recents</h3>
          <NavItem label="Change Python Version venv" />
          <NavItem label="Claude Code Plugin" />
          <NavItem label="Google Colab Notebook Help" />
          <NavItem label="Create venv guide" />
          <NavItem label="Fixing House Robber Code" />
        </div>
      </div>

      {/* Footer Profile */}
      <div className="p-3">
        <button className="w-full flex items-center gap-3 hover:bg-gray-800 p-2 rounded-md transition-colors">
          <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-semibold text-sm">
            IM
          </div>
          <div className="text-left">
            <div className="text-sm font-semibold">Ibrahim Mahidpur</div>
            <div className="text-xs text-gray-500">Go</div>
          </div>
        </button>
      </div>
    </div>
  );
};

const NavItem: React.FC<{ icon?: React.ReactNode; label: string }> = ({ icon, label }) => {
  return (
    <button className="w-full flex items-center gap-3 hover:bg-gray-800 p-2 rounded-md transition-colors text-gray-300">
      {icon ? <span className="text-gray-400">{icon}</span> : <MessageSquare size={16} className="text-gray-500" />}
      <span className="truncate">{label}</span>
    </button>
  );
};

export default Sidebar;
