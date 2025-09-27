import React, { useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
    Play, ChevronDown, CheckCircle, Linkedin,
    Camera, Target, TrendingUp, Shield,
} from "lucide-react";

/** Tiny placeholder avatar (initials + gradient) */
function Avatar({ name = "Member" }) {
    const initials = name.split(/\s+/).map(p => p[0]?.toUpperCase() || "").slice(0, 2).join("") || "??";
    return (
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-cyan-600 to-indigo-600
                        flex items-center justify-center text-white font-bold text-xl select-none">
            {initials}
        </div>
    );
}

const LandingPage = () => {
    const appRef = useRef(null);
    const aboutUsRef = useRef(null);
    const navigate = useNavigate();

    const scrollToSection = (ref) => ref.current?.scrollIntoView({ behavior: "smooth" });
    const handleHeadToApp = () => navigate("/app");

  /** Feature cards (icons only; copy is placeholder-friendly) */
    const features = [
        { icon: <Camera className="w-8 h-8" />, title: "Real-time Analysis",
        description: "High-FPS visuals rendered directly in the browser." },
        { icon: <Target className="w-8 h-8" />, title: "Form Correction",
        description: "Interactive controls to refine viewpoint and scale." },
        { icon: <TrendingUp className="w-8 h-8" />, title: "Progress Tracking",
        description: "Save and compare multiple generations quickly." },
        { icon: <Shield className="w-8 h-8" />, title: "Stability",
        description: "Graceful fallbacks and safe timeouts under load." },
    ];

  /** Team list WITHOUT images — just names/roles for now */
    const aboutUs = [
        { title: "Member One", role: "Fullstack Developer", linkedin: "" },
        { title: "Member Two", role: "Frontend Developer", linkedin: "" },
        { title: "Member Three", role: "Backend Developer", linkedin: "" },
        { title: "Member Four", role: "ML Engineer", linkedin: "" },
    ];

    return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-800 text-white">
      {/* ===== Nav ===== */}
        <nav className="fixed top-0 w-full bg-gray-950/90 backdrop-blur-sm z-50 border-b border-gray-700/50">
        <div className="relative flex items-center h-20 px-8 max-w-7xl mx-auto">
          {/* Brand */}
            <div className="absolute left-8">
            <button
                onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
                className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-indigo-500 bg-clip-text text-transparent
                            hover:from-cyan-300 hover:to-indigo-400 transition-all duration-300"
            >
                imagicle
            </button>
            </div>

          {/* Center links */}
            <div className="flex-1 flex justify-center space-x-16">
            <button
                onClick={() => scrollToSection(appRef)}
                className="text-gray-300 hover:text-cyan-400 transition-colors duration-300 font-medium"
            >
                What is imagicle?
            </button>
            <button
                onClick={() => scrollToSection(aboutUsRef)}
                className="text-gray-300 hover:text-cyan-400 transition-colors duration-300 font-medium"
            >
                About Us
            </button>
            </div>

          {/* CTA */}
            <div className="absolute right-8">
            <button
                onClick={handleHeadToApp}
                className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-cyan-500 text-white rounded-full font-semibold
                            transition-all duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-cyan-500/30"
            >
                Open App
            </button>
            </div>
        </div>
        </nav>

      {/* ===== Hero ===== */}
        <section ref={appRef} className="pt-36 pb-24 px-8 bg-gradient-to-b from-black/50 to-gray-900/30">
        <div className="max-w-7xl mx-auto text-center">
            <div className="mb-16">
            <h1 className="text-6xl md:text-7xl font-bold mb-8 bg-gradient-to-r from-white via-gray-200 to-cyan-400
                            bg-clip-text text-transparent leading-tight">
                Speak & See 3D
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed mb-4">
                Text or voice → generated 3D particles. Fast previews, upgrade to high fidelity when ready.
            </p>
            </div>

          {/* CTAs */}
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-20">
            <Link
                to="/app"
                className="px-10 py-5 bg-gradient-to-r from-indigo-600 to-cyan-500 text-white rounded-full text-lg font-semibold
                            transition-all duration-300 transform hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/20
                            flex items-center gap-3"
            >
                <Play className="w-5 h-5" />
                Try It Now
            </Link>

            <button
                onClick={() => scrollToSection(aboutUsRef)}
                className="px-10 py-5 border-2 border-cyan-500 text-cyan-300 rounded-full text-lg font-semibold
                            hover:bg-cyan-500/10 transition-all duration-300 flex items-center gap-3"
            >
                Learn About Us
                <ChevronDown className="w-5 h-5" />
            </button>
            </div>

          {/* Highlights */}
            <div className="flex flex-wrap justify-center items-center gap-8 text-sm text-gray-400 mb-20">
            {["Real-time preview", "Text & voice input", "Exportable results"].map((text, i) => (
                <span key={i} className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-cyan-400" />
                {text}
                </span>
            ))}
            </div>

          {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, idx) => (
                <div
                key={idx}
                className="group bg-gradient-to-br from-gray-800/80 to-black/60 rounded-2xl p-8 border border-gray-600/30
                            hover:border-cyan-400/40 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/10"
                >
                <div className="text-cyan-400 mb-6 group-hover:text-cyan-300 transition-colors">
                    {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-4 text-white group-hover:text-cyan-200 transition-colors">
                    {feature.title}
                </h3>
                <p className="text-gray-400 group-hover:text-gray-300 transition-colors leading-relaxed">
                    {feature.description}
                </p>
                </div>
            ))}
            </div>
        </div>
        </section>

      {/* ===== About Us (with placeholder avatars) ===== */}
        <section ref={aboutUsRef} className="relative py-20 px-8 bg-black overflow-hidden">
        {/* Decorative grid */}
        <div className="absolute inset-0">
            <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.03)_1px,transparent_1px)]
                            bg-[size:50px_50px]" />
            <div className="absolute top-0 left-1/4 w-px h-full bg-gradient-to-b from-transparent via-blue-500/20 to-transparent" />
            <div className="absolute top-0 right-1/3 w-px h-full bg-gradient-to-b from-transparent via-cyan-400/15 to-transparent" />
        </div>

        <div className="relative max-w-6xl mx-auto">
            <div className="mb-16 text-center">
            <h2 className="text-5xl md:text-6xl font-bold mb-8 bg-gradient-to-r from-white via-gray-200 to-cyan-400
                            bg-clip-text text-transparent leading-tight">
                Meet Our Team
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
                The builders behind <span className="font-semibold">imagicle</span>, crafting fast, beautiful 3D from text.
            </p>
            </div>

          {/* Team cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
            {aboutUs.map((member, index) => (
                <div
                key={index}
                className="group relative border border-slate-800 bg-slate-900/20 hover:border-cyan-400/40
                            transition-all duration-500 hover:-translate-y-1"
                >
                {/* index badge */}
                <div className="absolute top-4 right-4 z-10">
                    <span className="text-2xl font-thin text-slate-700 group-hover:text-slate-600 transition-colors duration-500 select-none">
                    {String(index + 1).padStart(2, "0")}
                    </span>
                </div>

                <div className="flex">
                  {/* Placeholder avatar column */}
                    <div className="w-1/2 bg-slate-600/60 flex items-center justify-center text-white p-8">
                    <div className="text-center">
                      {/* TODO: replace <Avatar /> with <img src={...} /> later */}
                        <Avatar name={member.title} />
                        <div className="text-xs uppercase tracking-wider opacity-80 mt-3">
                        {member.title}
                        </div>
                    </div>
                    </div>

                  {/* Content column */}
                    <div className="w-1/2 p-6 flex flex-col justify-center space-y-4">
                    <h3 className="text-lg font-bold text-white group-hover:text-cyan-100 transition-colors duration-300 uppercase tracking-wide">
                        {member.title}
                    </h3>
                    <p className="text-gray-400 font-medium tracking-wide text-sm">
                        {member.role}
                    </p>

                    {/* Placeholder LinkedIn (render only if provided later) */}
                    {member.linkedin ? (
                        <div className="pt-2">
                        <a
                            href={member.linkedin}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center text-xs text-gray-500 hover:text-cyan-400
                                        transition-colors duration-300 uppercase tracking-wider font-medium"
                        >
                            <Linkedin className="h-3 w-3 mr-2" />
                            LinkedIn
                        </a>
                        </div>
                    ) : (
                        <div className="pt-2 text-xs text-gray-500 italic">LinkedIn coming soon</div>
                    )}
                    </div>
                </div>

                <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-slate-700 to-transparent
                                group-hover:via-cyan-400/40 transition-all duration-500" />
                </div>
            ))}
            </div>

          {/* Accent */}
            <div className="mt-20 flex items-center justify-center">
            <div className="flex items-center gap-4">
                <div className="w-8 h-px bg-gradient-to-r from-transparent to-slate-600" />
                <div className="w-2 h-2 rounded-full bg-cyan-400/60" />
                <div className="w-8 h-px bg-gradient-to-l from-transparent to-slate-600" />
            </div>
            </div>
        </div>
        </section>
    </div>
    );
};

export default LandingPage;
