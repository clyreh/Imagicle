import React, { useState } from "react";
import { Play } from "lucide-react";

const ImageGeneratorPage = () => {
  const [prompt, setPrompt] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    setImageUrl("");

    try {
      // Replace with your backend API endpoint
      const response = await fetch("/api/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();
      setImageUrl(data.imageUrl);
    } catch (err) {
      console.error(err);
      alert("Error generating image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-800 text-white flex flex-col items-center px-6 py-16">
      {/* Title */}
      <h1 className="text-5xl md:text-6xl font-bold mb-8 text-center bg-gradient-to-r from-white via-gray-200 to-cyan-400 bg-clip-text text-transparent leading-tight">
        Generate Your 3D Image
      </h1>
      <p className="text-gray-300 text-lg md:text-xl max-w-3xl text-center mb-12">
        Type a prompt below and see your idea come to life as a generated image.
      </p>

      {/* Input + Button */}
      <div className="flex flex-col sm:flex-row gap-4 w-full max-w-3xl mb-12">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
          className="flex-1 px-4 py-3 rounded-full border border-gray-600 bg-gray-900 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-400 transition"
        />
        <button
          onClick={handleGenerate}
          className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-cyan-500 text-white rounded-full font-semibold text-lg hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/30 transition-all flex items-center justify-center gap-2"
        >
          <Play className="w-5 h-5" />
          Generate
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <p className="text-gray-400 text-center text-lg mb-6 animate-pulse">
          Generating image...
        </p>
      )}

      {/* Generated image */}
      {imageUrl && (
        <div className="w-full max-w-3xl flex justify-center">
          <img
            src={imageUrl}
            alt="Generated"
            className="rounded-2xl shadow-xl border border-gray-700"
          />
        </div>
      )}
    </div>
  );
};

export default ImageGeneratorPage;
