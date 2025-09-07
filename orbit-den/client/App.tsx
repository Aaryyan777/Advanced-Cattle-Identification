import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <div className="min-h-screen grid grid-rows-[auto,1fr,auto]">
          <header className="border-b-4 border-foreground bg-primary text-primary-foreground">
            <div className="container mx-auto py-3 flex items-center justify-between">
              <a href="/" className="text-2xl font-extrabold uppercase tracking-tight border-4 border-foreground px-2 bg-background text-foreground">CattleScope</a>
              <nav className="flex items-center gap-4 text-sm uppercase">
                <a href="/#how" className="underline-offset-4 hover:underline">How it works</a>
                <a href="/#about" className="underline-offset-4 hover:underline">About</a>
              </nav>
            </div>
          </header>
          <main className="container mx-auto py-8">
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>
          <footer className="border-t-4 border-foreground bg-secondary">
            <div className="container mx-auto py-4 text-xs flex items-center justify-between">
              <span>© {new Date().getFullYear()} CattleScope</span>
              <span className="uppercase">Built for precise breed ID</span>
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
