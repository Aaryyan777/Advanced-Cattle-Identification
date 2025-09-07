export type BreedInfo = {
  name: string;
  origin: string;
  purpose: string[];
  description: string;
  images: { url: string; credit?: string }[];
  colors?: string[];
  size?: "small" | "medium" | "large";
};

export const BREEDS: Record<string, BreedInfo> = {
  Holstein: {
    name: "Holstein",
    origin: "Netherlands",
    purpose: ["Dairy"],
    description:
      "Holsteins are easily recognized by their distinctive black-and-white spotted pattern and are the world's highest-producing dairy breed.",
    images: [
      {
        url: "https://upload.wikimedia.org/wikipedia/commons/2/2e/Holstein_Friesian_cow.jpg",
        credit: "Wikimedia Commons",
      },
      {
        url: "https://upload.wikimedia.org/wikipedia/commons/7/7d/Holstein_cow.jpg",
      },
    ],
    colors: ["black", "white"],
    size: "large",
  },
  Angus: {
    name: "Angus",
    origin: "Scotland",
    purpose: ["Beef"],
    description:
      "Angus cattle are typically solid black (or red in Red Angus), known for excellent marbling and beef quality.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/3/3d/Black_Angus_cow.jpg" },
      { url: "https://upload.wikimedia.org/wikipedia/commons/9/95/Angus_cattle.jpg" },
    ],
    colors: ["black"],
    size: "medium",
  },
  Hereford: {
    name: "Hereford",
    origin: "England",
    purpose: ["Beef"],
    description:
      "Herefords have a red-brown body with a white face, crest, underline, and often white on the legs and switch.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/6/6f/Hereford_cow.jpg" },
      { url: "https://upload.wikimedia.org/wikipedia/commons/2/2a/Hereford_bull.jpg" },
    ],
    colors: ["red", "white"],
    size: "large",
  },
  "Brown Swiss": {
    name: "Brown Swiss",
    origin: "Switzerland",
    purpose: ["Dairy"],
    description:
      "Brown Swiss are sturdy dairy cattle with a solid brown coat, renowned for milk with high butterfat and protein.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/e/e9/Brown_Swiss_cow.jpg" },
    ],
    colors: ["brown", "grey"],
    size: "large",
  },
  Jersey: {
    name: "Jersey",
    origin: "Jersey (Channel Islands)",
    purpose: ["Dairy"],
    description:
      "A smaller dairy breed with light brown to fawn coloring, Jerseys are known for rich, high-butterfat milk and docile nature.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/8/8b/Jersey_cow.jpg" },
    ],
    colors: ["fawn", "light brown"],
    size: "small",
  },
  Brahman: {
    name: "Brahman",
    origin: "India",
    purpose: ["Beef", "Draft"],
    description:
      "Brahman cattle have a distinctive hump over the shoulders, long ears, and loose skin. They are heat-tolerant and hardy.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/3/34/Brahman_cow.jpg" },
    ],
    colors: ["grey", "red"],
    size: "large",
  },
  Charolais: {
    name: "Charolais",
    origin: "France",
    purpose: ["Beef"],
    description:
      "Charolais are large, white to cream-colored cattle prized for muscularity and growth rate.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/5/5e/Charolais_cattle.jpg" },
    ],
    colors: ["white", "cream"],
    size: "large",
  },
  Simmental: {
    name: "Simmental",
    origin: "Switzerland",
    purpose: ["Dual-purpose"],
    description:
      "Simmentals are often red and white with a white face, used for both milk and beef with strong growth and adaptability.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/4/40/Simmental_cow.jpg" },
    ],
    colors: ["red", "white"],
    size: "large",
  },
  Guernsey: {
    name: "Guernsey",
    origin: "Guernsey (Channel Islands)",
    purpose: ["Dairy"],
    description:
      "Guernseys are a dairy breed with fawn to red coats and white markings, producing rich golden-colored milk.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/6/60/Guernsey_cow.jpg" },
    ],
    colors: ["fawn", "red", "white"],
    size: "medium",
  },
  Ayrshire: {
    name: "Ayrshire",
    origin: "Scotland",
    purpose: ["Dairy"],
    description:
      "Ayrshires are red and white dairy cattle known for vigor and efficiency, with strong udder conformation.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/4/4f/Ayrshire_cow.jpg" },
    ],
    colors: ["red", "white"],
    size: "medium",
  },
  Gyr: {
    name: "Gyr",
    origin: "India",
    purpose: ["Dairy"],
    description:
      "Gyr (Gir) are zebu cattle with distinctive long, pendulous ears and a convex forehead, adapted to heat with good milk yields.",
    images: [
      { url: "https://upload.wikimedia.org/wikipedia/commons/3/37/Gir_cow.jpg" },
    ],
    colors: ["red", "white", "spotted"],
    size: "medium",
  },
};

export function normalizeBreedName(name: string): string {
  const key = Object.keys(BREEDS).find(
    (k) => k.toLowerCase() === name.toLowerCase(),
  );
  return key ?? name;
}
