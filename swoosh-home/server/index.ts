import "dotenv/config";
import express from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import { handleClassify } from "./routes/classify";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  const jsonLimit = process.env.JSON_LIMIT ?? "50mb";
  app.use(express.json({ limit: jsonLimit }));
  app.use(express.urlencoded({ extended: true, limit: jsonLimit }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    const ping = process.env.PING_MESSAGE ?? "ping";
    res.json({ message: ping });
  });

  app.get("/api/demo", handleDemo);

  // ML Inference proxy
  app.post("/api/classify", handleClassify);

  return app;
}
