#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>

enum class TokenType {
	plus, minus, mult, div, pow,
	lpar, rpar, lbrace, rbrace,
	symbol, integer, real,
	derivative, partial, integral,
	covariant, contravariant, transpose,
	inner, outer,
	end, unknown
};

enum class GreekSymbolminus {
	alpha, beta, gamma, delta, epsilon,
	zeta, eta, theta, iota, kappa,
	lambda, mu, nu, xi, omicron,
	pi, rho, sigma, tau, upsilon,
	phi, chi, psi, omega
};

enum class GreekSymbolplus {
	Alpha, Beta, Gamma, Delta, Epsilon,
	Zeta, Eta, Theta, Iota, Kappa,
	Lambda, Mu, Nu, Xi, Omicron,
	Pi, Rho, Sigma, Tau, Upsilon,
	Phi, Chi, Psi, Omega
};


struct Token {
	TokenType type;
	GreekSymbolminus greek_minus = GreekSymbolminus::alpha;
	GreekSymbolplus greek_plus = GreekSymbolplus::Alpha;
	std::string value;

	Token(TokenType t, const std::string& val)
		: type(t), value(val) {}

	Token(TokenType t, GreekSymbolminus g, GreekSymbolplus G, const std::string& val)
		: type(t), greek_minus(g), greek_plus(G), value(val) {}
};


class Lexer {
	public:
		explicit Lexer(const std::string& input) : input(input), pos(0) {}

		std::vector<Token> tokenize() {
			std::vector<Token> tokens;

			while (!eof()) {
				skipWhitespace();
				char c = peek();

				if (std::isdigit(c)) {
					tokens.push_back(parseNumber());
				} else if (c == '\\' || std::isalpha(c)) {
					tokens.push_back(parseCommandOrSymbol());
				} else {
					std::string s(1, get());
					auto it = SyntaxTable.find(s);
					tokens.push_back({
							it != SyntaxTable.end() ? it->second : TokenType::unknown,
							s
							});
				}
			}

			tokens.emplace_back(TokenType::end, "");
			return tokens;
		}

	private:
		std::string input;
		size_t pos;

		char peek() const {
			return pos < input.size() ? input[pos] : '\0';
		}

		char get() {
			return pos < input.size() ? input[pos++] : '\0';
		}

		bool eof() const {
			return pos >= input.size();
		}

		void skipWhitespace() {
			while (!eof() && std::isspace(peek()))
				get();
		}

		Token parseNumber() {
			std::string num;
			bool has_dot = false;

			while (!eof() && (std::isdigit(peek()) || peek() == '.')) {
				if (peek() == '.') has_dot = true;
				num += get();
			}

			return {
				has_dot ? TokenType::real : TokenType::integer,
						num
			};
		}

		Token parseCommandOrSymbol() {
			std::string s;
			if (peek() == '\\') s += get(); 

			while (!eof() && (std::isalpha(peek()) || peek() == '_')) {
				s += get();
			}

			if (auto gmin = GreekMapLower.find(s); gmin != GreekMapLower.end()) {
				return Token(TokenType::symbol, gmin->second, GreekSymbolplus::Alpha, s);
			}

			if (auto gmaj = GreekMapUpper.find(s); gmaj != GreekMapUpper.end()) {
				return Token(TokenType::symbol, GreekSymbolminus::alpha, gmaj->second, s);
			}

			if (auto tok = SyntaxTable.find(s); tok != SyntaxTable.end()) {
				return Token(tok->second, s);
			}

			return Token(TokenType::symbol, s);
		}

	private:
		inline static const std::unordered_map<std::string, TokenType> SyntaxTable = {
			{"+", TokenType::plus}, {"-", TokenType::minus}, {"*", TokenType::mult}, {"/", TokenType::div}, {"^", TokenType::pow},
			{"(", TokenType::lpar}, {")", TokenType::rpar}, {"{", TokenType::lbrace}, {"}", TokenType::rbrace},
			{"d", TokenType::derivative}, {"\\partial", TokenType::partial}, {"\\int", TokenType::integral},
			{"cov", TokenType::covariant}, {"contr", TokenType::contravariant}, {"T", TokenType::transpose},
			{"\\cdot", TokenType::inner}, {"\\otimes", TokenType::outer},
		};

		inline static const std::unordered_map<std::string, GreekSymbolminus> GreekMapLower = {
			{"\\alpha", GreekSymbolminus::alpha}, {"\\beta", GreekSymbolminus::beta},
			{"\\gamma", GreekSymbolminus::gamma}, {"\\delta", GreekSymbolminus::delta},
			{"\\epsilon", GreekSymbolminus::epsilon}, {"\\zeta", GreekSymbolminus::zeta},
			{"\\eta", GreekSymbolminus::eta}, {"\\theta", GreekSymbolminus::theta},
			{"\\iota", GreekSymbolminus::iota}, {"\\kappa", GreekSymbolminus::kappa},
			{"\\lambda", GreekSymbolminus::lambda}, {"\\mu", GreekSymbolminus::mu},
			{"\\nu", GreekSymbolminus::nu}, {"\\xi", GreekSymbolminus::xi},
			{"\\omicron", GreekSymbolminus::omicron}, {"\\pi", GreekSymbolminus::pi},
			{"\\rho", GreekSymbolminus::rho}, {"\\sigma", GreekSymbolminus::sigma},
			{"\\tau", GreekSymbolminus::tau}, {"\\upsilon", GreekSymbolminus::upsilon},
			{"\\phi", GreekSymbolminus::phi}, {"\\chi", GreekSymbolminus::chi},
			{"\\psi", GreekSymbolminus::psi}, {"\\omega", GreekSymbolminus::omega}
		};

		inline static const std::unordered_map<std::string, GreekSymbolplus> GreekMapUpper = {
			{"\\Alpha", GreekSymbolplus::Alpha}, {"\\Beta", GreekSymbolplus::Beta},
			{"\\Gamma", GreekSymbolplus::Gamma}, {"\\Delta", GreekSymbolplus::Delta},
			{"\\Epsilon", GreekSymbolplus::Epsilon}, {"\\Zeta", GreekSymbolplus::Zeta},
			{"\\Eta", GreekSymbolplus::Eta}, {"\\Theta", GreekSymbolplus::Theta},
			{"\\Iota", GreekSymbolplus::Iota}, {"\\Kappa", GreekSymbolplus::Kappa},
			{"\\Lambda", GreekSymbolplus::Lambda}, {"\\Mu", GreekSymbolplus::Mu},
			{"\\Nu", GreekSymbolplus::Nu}, {"\\Xi", GreekSymbolplus::Xi},
			{"\\Omicron", GreekSymbolplus::Omicron}, {"\\Pi", GreekSymbolplus::Pi},
			{"\\Rho", GreekSymbolplus::Rho}, {"\\Sigma", GreekSymbolplus::Sigma},
			{"\\Tau", GreekSymbolplus::Tau}, {"\\Upsilon", GreekSymbolplus::Upsilon},
			{"\\Phi", GreekSymbolplus::Phi}, {"\\Chi", GreekSymbolplus::Chi},
			{"\\Psi", GreekSymbolplus::Psi}, {"\\Omega", GreekSymbolplus::Omega}
		};

};
