import React, { useState, useEffect } from "react";

function TypewriterWord({ word = "Type", speed = 150, className = "" }) {
  const [shown, setShown] = useState("");

  useEffect(() => {
    let i = 0;
    const id = setInterval(() => {
      setShown(word.slice(0, i + 1)); // reveal one more letter
      i += 1;
      if (i === word.length) clearInterval(id); // stop at full word
    }, speed);

    return () => clearInterval(id); // cleanup if unmounts
  }, [word, speed]);

  return (
    <span className={className}>
      {shown}
      <span className="animate-pulse">|</span>
    </span>
  );
}

export default TypewriterWord;
