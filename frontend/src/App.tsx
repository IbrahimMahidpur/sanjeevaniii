import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";

const App: React.FC = () => {
  const [language, setLanguage] = useState<string>("en");

  return (
    <div className="h-screen w-full bg-gpt-main text-gray-200 flex font-sans overflow-hidden">
      <main className="flex-1 relative flex flex-col min-w-0">
        <div className="absolute top-4 left-4 z-10">
          <button className="text-gray-400 hover:text-white flex items-center gap-2 font-semibold text-lg px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors">
            Sanjeevanii <span className="text-gray-500 text-sm">v</span>
          </button>
        </div>
        <div className="flex-1 overflow-hidden">
           <ChatWindow language={language} />
        </div>
      </main>
    </div>
  );
};

export default App;
