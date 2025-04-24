#pragma once 

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include "Latex.hpp"

namespace morpheus {

	enum class ASTNodeType {
		Number,
		Symbol,
		BinaryOp,
		UnaryOp,
		FunctionCall,
		Subscript,
		Superscript,
		Derivative,
		Integral,
		TensorSymbol 
	};

	struct ASTNode {
		ASTNodeType type;
		std::string value;
		std::vector<std::shared_ptr<ASTNode>> children;

		ASTNode(ASTNodeType type, const std::string& val = "")
			: type(type), value(val) {}

		ASTNode(ASTNodeType type, const std::string& val, const std::vector<std::shared_ptr<ASTNode>>& childs)
			: type(type), value(val), children(childs) {}
	};

	class Parser {
		public:
			explicit Parser(const std::vector<Token>& tokens) : tokens(tokens), pos(0) {}

			std::shared_ptr<ASTNode> parse_expression();
			std::shared_ptr<ASTNode> parse_expression(int precedence = 0) {
				auto lhs = parse_primary();
				if (!lhs) return nullptr;
				return parse_binary_rhs(precedence, lhs);
			}

		private:
			std::vector<Token> tokens;
			size_t pos;

			std::shared_ptr<ASTNode> parse_primary();
			std::shared_ptr<ASTNode> parse_binary_rhs(int prec, std::shared_ptr<ASTNode> lhs);

			Token peek() const { return tokens[pos]; }
			Token get() { return tokens[pos++]; }
			bool eof() const { return pos >= tokens.size(); }

			int get_precedence(TokenType t) const {
				switch (t) {
					case TokenType::plus:
					case TokenType::minus:
						return 10;
					case TokenType::mult:
					case TokenType::div:
						return 20;
					case TokenType::pow:
						return 30;
					default:
						return -1;
				}
			}
	};

	inline void print_ast(const std::shared_ptr<ASTNode>& node, int indent = 0) {
		if (!node) return;
		std::string pad(indent, ' ');
		std::cout << pad << "Node: " << node->value << " [Type=" << static_cast<int>(node->type) << "]\n";
		for (const auto& child : node->children)
			print_ast(child, indent + 2);
	}

} 
