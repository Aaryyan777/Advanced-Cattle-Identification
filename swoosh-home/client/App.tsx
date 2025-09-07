import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import { Button } from "@/components/ui/button";

const queryClient = new QueryClient();

const Layout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen bg-background text-foreground relative">
    <div className="absolute inset-0 bg-grid pointer-events-none" aria-hidden="true" />
    <header className="relative z-10 border-b-4 border-foreground bg-background">
      <div className="container h-16 flex items-center justify-between">
        <Link to="/" className="font-extrabold text-xl tracking-tight">
          CATTLE.VISION
        </Link>
        <div className="flex items-center gap-2">
          <Button
            asChild
            className="border-4 border-foreground bg-primary text-primary-foreground brutal-shadow"
          >
            <a href="#uploader">Try it now</a>
          </Button>
        </div>
      </div>
    </header>

    <main className="relative z-10">{children}</main>

    <footer className="relative z-10 mt-16 border-t-4 border-foreground">
      <div className="container py-6 text-xs text-muted-foreground flex items-center justify-between">
        <p>Built for cattle breed identification • Brutalist UI</p>
        <p>
          <a href="https://builder.io/c/docs/projects" className="underline" target="_blank" rel="noreferrer">
            Docs
          </a>
        </p>
      </div>
    </footer>
  </div>
);

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Index />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
