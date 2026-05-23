"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="default">
      <div className="flex flex-col flex-1 items-center justify-center h-screen bg-zinc-50 dark:bg-black w-full">
        <CopilotChat
          className="w-full max-w-3xl h-full"
          labels={{ title: "AI Assistant", initial: "Hi! How can I help you today?" }}
        />
      </div>
    </CopilotKit>
  );
}
