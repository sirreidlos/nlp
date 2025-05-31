{
  description = "Python Packages Example";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python3.override {
          self = python;
          packageOverrides = self: super: {
            protobuf = super.protobuf.overrideAttrs (old: {
              version = "3.20.3";

              src = pkgs.fetchPypi {
                pname = "protobuf";
                version = "3.20.3";
                hash =
                  "sha256-N9jQ9dJ/N+fKMeWsNYFfR3jRrgHo0rYPvWb0hcrEVWg="; # You can use `nix-prefetch pypi protobuf 3.20.3` to get the correct hash.
              };
            });
          };
        };

      in with pkgs; {
        devShells.default = mkShell {
          name = "rm";
          packages = [
            # put any non-Python packages here
            # google-cloud-sdk
            # Python packages:
            (python.withPackages (p:
              with p; [
                glob2
                gtts
                ipython
                jupyter
                librosa
                matplotlib
                nbconvert
                numpy
                pandas
                pip
                pyarrow
                pyaudio
                pydub
                scikit-learn
                scipy
                seaborn
                sentencepiece
                sounddevice
                soundfile
                tensorflow
                tensorflow-datasets
                torch
                torchaudio
                torch-audiomentations
                transformers
                tqdm
                tf-keras
                accelerate
                levenshtein
                nltk
                evaluate

                streamlit
              ]))
          ];

          shellHook = ''
            echo "Look at all these amazing Python packages!"
          '';
        };
      });
}
